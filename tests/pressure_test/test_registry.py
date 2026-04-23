"""Registry pressure test: persist and recover artifacts, runs, checks, lineage.

Tests whether contracts serialize cleanly through DuckDB and come back identical.

Failure signals:
- Artifact fields lost or mangled during round-trip
- RunRecord losing output/checks association
- Check truth getting confused between artifact and run ownership
- Lineage edges not forming a coherent graph
- Backing store becoming illegible after deserialization
"""

import numpy as np
import pytest
import rasterio
from quarry_connectors.local_file import LocalFileConnector
from quarry_core.artifact import ArtifactType, BackingStoreKind, ValidationState
from quarry_core.check import CRSValid
from quarry_core.executor import RunStatus
from quarry_core.executors.local import LocalExecutor
from quarry_operators.clip_raster import ClipRasterOperator, ClipRasterParams
from quarry_registry.registry import Registry
from rasterio.crs import CRS
from rasterio.transform import from_bounds


@pytest.fixture
def sample_raster(tmp_path):
    """Create a small synthetic raster."""
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
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ws


@pytest.fixture
def registry(workspace):
    return Registry(workspace)


class TestArtifactRoundTrip:
    """Artifact persists and recovers losslessly."""

    def test_save_and_get(self, sample_raster, workspace, registry):
        conn = LocalFileConnector()
        mat = conn.materialize(str(sample_raster), workspace)
        artifact = mat.artifact

        registry.save_artifact(artifact)
        recovered = registry.get_artifact(artifact.id)

        assert recovered is not None
        assert recovered.id == artifact.id
        assert recovered.type == artifact.type
        assert recovered.name == artifact.name

    def test_backing_store_round_trip(self, sample_raster, workspace, registry):
        conn = LocalFileConnector()
        mat = conn.materialize(str(sample_raster), workspace)
        artifact = mat.artifact

        registry.save_artifact(artifact)
        recovered = registry.get_artifact(artifact.id)

        assert recovered.backing is not None
        assert recovered.backing.kind == artifact.backing.kind
        assert recovered.backing.uri == artifact.backing.uri
        assert recovered.backing.size_bytes == artifact.backing.size_bytes
        assert recovered.backing.content_hash == artifact.backing.content_hash

    def test_spatial_round_trip(self, sample_raster, workspace, registry):
        conn = LocalFileConnector()
        mat = conn.materialize(str(sample_raster), workspace)
        artifact = mat.artifact

        registry.save_artifact(artifact)
        recovered = registry.get_artifact(artifact.id)

        assert recovered.spatial.crs == artifact.spatial.crs
        assert recovered.spatial.extent is not None
        for i in range(4):
            assert abs(recovered.spatial.extent[i] - artifact.spatial.extent[i]) < 1e-10
        assert recovered.spatial.band_count == artifact.spatial.band_count
        assert recovered.spatial.resolution is not None
        assert abs(recovered.spatial.resolution[0] - artifact.spatial.resolution[0]) < 1e-10

    def test_metadata_round_trip(self, sample_raster, workspace, registry):
        conn = LocalFileConnector()
        mat = conn.materialize(str(sample_raster), workspace)
        artifact = mat.artifact

        registry.save_artifact(artifact)
        recovered = registry.get_artifact(artifact.id)

        assert recovered.metadata == artifact.metadata

    def test_with_check_does_not_alias_metadata(self, sample_raster, workspace):
        conn = LocalFileConnector()
        artifact = conn.materialize(str(sample_raster), workspace).artifact

        updated = artifact.with_check(
            CRSValid().run(artifact)
        )

        assert updated.metadata == artifact.metadata
        assert updated.metadata is not artifact.metadata

    def test_lineage_round_trip(self, sample_raster, workspace, registry):
        conn = LocalFileConnector()
        artifact = conn.materialize(str(sample_raster), workspace).artifact

        registry.save_artifact(artifact)
        recovered = registry.get_artifact(artifact.id)

        assert recovered.lineage is not None
        assert recovered.lineage.operation == artifact.lineage.operation
        assert dict(recovered.lineage.params) == dict(artifact.lineage.params)

    def test_lazy_artifact_round_trip(self, sample_raster, workspace, registry):
        conn = LocalFileConnector()
        mat = conn.materialize(str(sample_raster), workspace, lazy=True)
        artifact = mat.artifact

        registry.save_artifact(artifact)
        recovered = registry.get_artifact(artifact.id)

        assert recovered.backing.kind == BackingStoreKind.LAZY_HANDLE
        assert not recovered.is_materialized

    def test_list_artifacts(self, sample_raster, workspace, registry):
        conn = LocalFileConnector()

        # Save two artifacts
        a1 = conn.materialize(str(sample_raster), workspace).artifact
        a2 = conn.materialize(str(sample_raster), workspace).artifact
        registry.save_artifact(a1)
        registry.save_artifact(a2)

        all_arts = registry.list_artifacts()
        assert len(all_arts) == 2

        raster_arts = registry.list_artifacts(artifact_type=ArtifactType.RASTER)
        assert len(raster_arts) == 2

        vector_arts = registry.list_artifacts(artifact_type=ArtifactType.VECTOR)
        assert len(vector_arts) == 0

    def test_nonexistent_artifact_returns_none(self, registry):
        assert registry.get_artifact("nonexistent-id") is None


class TestRunRoundTrip:
    """RunRecord persists with output artifact, checks, and lineage."""

    def test_full_run_round_trip(self, sample_raster, workspace, registry):
        # Execute a real flow
        conn = LocalFileConnector()
        mat = conn.materialize(str(sample_raster), workspace)
        registry.save_artifact(mat.artifact)

        output_path = workspace / "clipped.tif"
        params = ClipRasterParams(
            bounds=(-5.0, -5.0, 5.0, 5.0),
            output_path=str(output_path),
        )
        executor = LocalExecutor()
        record = executor.submit(ClipRasterOperator(), [mat.artifact], params)

        # Save run (which cascades to output artifact + checks + lineage)
        registry.save_run(record)

        # Recover
        recovered = registry.get_run(record.id)
        assert recovered is not None
        assert recovered.id == record.id
        assert recovered.operator_name == "clip_raster"
        assert recovered.status == RunStatus.COMPLETED
        assert recovered.input_ids == [mat.artifact.id]
        assert recovered.executor_name == "local"
        assert recovered.started_at is not None
        assert recovered.completed_at is not None
        assert recovered.error is None

    def test_run_saves_output_artifact(self, sample_raster, workspace, registry):
        conn = LocalFileConnector()
        mat = conn.materialize(str(sample_raster), workspace)
        registry.save_artifact(mat.artifact)

        output_path = workspace / "clipped2.tif"
        params = ClipRasterParams(
            bounds=(-5.0, -5.0, 5.0, 5.0),
            output_path=str(output_path),
        )
        executor = LocalExecutor()
        record = executor.submit(ClipRasterOperator(), [mat.artifact], params)
        registry.save_run(record)

        # Output artifact should be in registry
        output_id = record.output.artifact.id
        output_art = registry.get_artifact(output_id)
        assert output_art is not None
        assert output_art.type == ArtifactType.RASTER

    def test_list_runs(self, sample_raster, workspace, registry):
        conn = LocalFileConnector()
        mat = conn.materialize(str(sample_raster), workspace)
        registry.save_artifact(mat.artifact)

        for i in range(3):
            output_path = workspace / f"run_{i}.tif"
            params = ClipRasterParams(
                bounds=(-5.0, -5.0, 5.0, 5.0),
                output_path=str(output_path),
            )
            record = LocalExecutor().submit(ClipRasterOperator(), [mat.artifact], params)
            registry.save_run(record)

        all_runs = registry.list_runs()
        assert len(all_runs) == 3

        completed = registry.list_runs(status=RunStatus.COMPLETED)
        assert len(completed) == 3

    def test_run_output_reconstructed(self, sample_raster, workspace, registry):
        """get_run() must return a RunRecord with output.artifact matching the original."""
        conn = LocalFileConnector()
        mat = conn.materialize(str(sample_raster), workspace)
        registry.save_artifact(mat.artifact)

        output_path = workspace / "output_rt.tif"
        params = ClipRasterParams(
            bounds=(-5.0, -5.0, 5.0, 5.0),
            output_path=str(output_path),
        )
        record = LocalExecutor().submit(ClipRasterOperator(), [mat.artifact], params)
        registry.save_run(record)

        recovered = registry.get_run(record.id)
        assert recovered.output is not None
        assert recovered.output.artifact.id == record.output.artifact.id
        assert recovered.output.artifact.type == record.output.artifact.type
        assert recovered.output.artifact.name == record.output.artifact.name

    def test_failed_run_has_no_output(self, sample_raster, workspace, registry):
        """A run that never produced output should have output=None after round-trip."""
        from quarry_core.executor import RunRecord

        run = RunRecord(
            id="failed-run-001",
            operator_name="clip_raster",
            status=RunStatus.FAILED,
            error="boom",
        )
        registry.save_run(run)

        recovered = registry.get_run("failed-run-001")
        assert recovered is not None
        assert recovered.output is None
        assert recovered.error == "boom"

    def test_run_output_checks_match(self, sample_raster, workspace, registry):
        """Checks on reconstructed output should match checks on the run."""
        conn = LocalFileConnector()
        mat = conn.materialize(str(sample_raster), workspace)
        registry.save_artifact(mat.artifact)

        output_path = workspace / "checks_rt.tif"
        params = ClipRasterParams(
            bounds=(-5.0, -5.0, 5.0, 5.0),
            output_path=str(output_path),
        )
        record = LocalExecutor().submit(ClipRasterOperator(), [mat.artifact], params)
        registry.save_run(record)

        recovered = registry.get_run(record.id)
        assert recovered.output is not None
        assert len(recovered.output.checks) == len(recovered.checks)
        for oc, rc in zip(recovered.output.checks, recovered.checks):
            assert oc.check_name == rc.check_name
            assert oc.state == rc.state

    def test_run_persists_output_checks_as_single_source_of_truth(
        self, sample_raster, workspace, registry
    ):
        conn = LocalFileConnector()
        mat = conn.materialize(str(sample_raster), workspace)
        registry.save_artifact(mat.artifact)

        output_path = workspace / "output_only_checks.tif"
        params = ClipRasterParams(
            bounds=(-5.0, -5.0, 5.0, 5.0),
            output_path=str(output_path),
        )
        record = LocalExecutor().submit(ClipRasterOperator(), [mat.artifact], params)
        registry.save_run(record)

        recovered = registry.get_run(record.id)
        assert recovered is not None
        assert len(recovered.output.checks) > 0
        assert len(recovered.checks) == len(recovered.output.checks)

    def test_save_run_replaces_existing_run_checks(self, sample_raster, workspace, registry):
        conn = LocalFileConnector()
        mat = conn.materialize(str(sample_raster), workspace)
        registry.save_artifact(mat.artifact)

        output_path = workspace / "idempotent_checks.tif"
        params = ClipRasterParams(
            bounds=(-5.0, -5.0, 5.0, 5.0),
            output_path=str(output_path),
        )
        record = LocalExecutor().submit(ClipRasterOperator(), [mat.artifact], params)

        registry.save_run(record)
        registry.save_run(record)

        recovered = registry.get_run(record.id)
        assert recovered is not None
        assert len(recovered.checks) == len(record.checks)

    def test_run_output_result_fields_round_trip(self, sample_raster, workspace, registry):
        """OperatorResult metadata survives registry round-trip."""
        conn = LocalFileConnector()
        mat = conn.materialize(str(sample_raster), workspace)
        registry.save_artifact(mat.artifact)

        output_path = workspace / "result_fields.tif"
        params = ClipRasterParams(
            bounds=(-5.0, -5.0, 5.0, 5.0),
            output_path=str(output_path),
        )
        record = LocalExecutor().submit(ClipRasterOperator(), [mat.artifact], params)

        # Inject ephemeral fields to test persistence
        record.output.warnings = ["test-warning"]
        record.output.metadata = {"test-key": "test-value"}

        registry.save_run(record)

        recovered = registry.get_run(record.id)
        assert recovered is not None
        assert recovered.output is not None
        assert recovered.output.timing_seconds is not None
        assert recovered.output.timing_seconds >= 0
        assert recovered.output.warnings == ["test-warning"]
        assert recovered.output.metadata == {"test-key": "test-value"}
        assert recovered.output.artifact.lineage is not None

    def test_list_runs_reconstructs_output(self, sample_raster, workspace, registry):
        """list_runs() should also reconstruct output on each RunRecord."""
        conn = LocalFileConnector()
        mat = conn.materialize(str(sample_raster), workspace)
        registry.save_artifact(mat.artifact)

        output_path = workspace / "list_rt.tif"
        params = ClipRasterParams(
            bounds=(-5.0, -5.0, 5.0, 5.0),
            output_path=str(output_path),
        )
        record = LocalExecutor().submit(ClipRasterOperator(), [mat.artifact], params)
        registry.save_run(record)

        runs = registry.list_runs()
        assert len(runs) == 1
        assert runs[0].output is not None
        assert runs[0].output.artifact.id == record.output.artifact.id

    def test_nonexistent_run_returns_none(self, registry):
        assert registry.get_run("nonexistent-id") is None


class TestCheckPersistence:
    """Check truth lives in the checks table, not embedded in artifacts."""

    def test_checks_saved_with_run(self, sample_raster, workspace, registry):
        conn = LocalFileConnector()
        mat = conn.materialize(str(sample_raster), workspace)
        registry.save_artifact(mat.artifact)

        output_path = workspace / "checked.tif"
        params = ClipRasterParams(
            bounds=(-5.0, -5.0, 5.0, 5.0),
            output_path=str(output_path),
        )
        record = LocalExecutor().submit(ClipRasterOperator(), [mat.artifact], params)
        registry.save_run(record)

        # Query checks by artifact
        output_id = record.output.artifact.id
        artifact_checks = registry.get_checks(artifact_id=output_id)
        assert len(artifact_checks) > 0

        # Query checks by run
        run_checks = registry.get_checks(run_id=record.id)
        assert len(run_checks) > 0

        # Both should be the same checks
        assert len(artifact_checks) == len(run_checks)

    def test_checks_loaded_with_artifact(self, sample_raster, workspace, registry):
        """Artifact loaded from registry carries its checks."""
        conn = LocalFileConnector()
        mat = conn.materialize(str(sample_raster), workspace)
        registry.save_artifact(mat.artifact)

        output_path = workspace / "with_checks.tif"
        params = ClipRasterParams(
            bounds=(-5.0, -5.0, 5.0, 5.0),
            output_path=str(output_path),
        )
        record = LocalExecutor().submit(ClipRasterOperator(), [mat.artifact], params)
        registry.save_run(record)

        # Load artifact — should have checks from the checks table
        output_id = record.output.artifact.id
        recovered = registry.get_artifact(output_id)
        assert recovered.validation_state != ValidationState.UNCHECKED
        assert len(recovered.checks) > 0

    def test_independent_check_save(self, sample_raster, workspace, registry):
        """Checks can be saved independently of runs."""
        conn = LocalFileConnector()
        mat = conn.materialize(str(sample_raster), workspace)
        registry.save_artifact(mat.artifact)

        # Run a check independently
        check_result = CRSValid().run(mat.artifact)
        registry.save_check(artifact_id=mat.artifact.id, check=check_result)

        # Retrieve
        checks = registry.get_checks(artifact_id=mat.artifact.id)
        assert len(checks) == 1
        assert checks[0].check_name == "crs_valid"
        assert checks[0].state == ValidationState.VALID


class TestLineage:
    """Lineage edges form a coherent graph."""

    def test_parent_child_edges(self, sample_raster, workspace, registry):
        conn = LocalFileConnector()
        mat = conn.materialize(str(sample_raster), workspace)
        registry.save_artifact(mat.artifact)

        output_path = workspace / "child.tif"
        params = ClipRasterParams(
            bounds=(-5.0, -5.0, 5.0, 5.0),
            output_path=str(output_path),
        )
        record = LocalExecutor().submit(ClipRasterOperator(), [mat.artifact], params)
        registry.save_run(record)

        output_id = record.output.artifact.id

        # Parent → child
        children = registry.get_children(mat.artifact.id)
        assert len(children) == 1
        assert children[0]["artifact_id"] == output_id
        assert children[0]["operation"] == "clip_raster"

        # Child → parent
        parents = registry.get_parents(output_id)
        assert len(parents) == 1
        assert parents[0]["artifact_id"] == mat.artifact.id

    def test_multi_generation_lineage(self, sample_raster, workspace, registry):
        """Three-generation chain: source → clip → clip again."""
        conn = LocalFileConnector()
        mat = conn.materialize(str(sample_raster), workspace)
        registry.save_artifact(mat.artifact)

        # Generation 1: clip
        out1 = workspace / "gen1.tif"
        params1 = ClipRasterParams(bounds=(-5.0, -5.0, 5.0, 5.0), output_path=str(out1))
        record1 = LocalExecutor().submit(ClipRasterOperator(), [mat.artifact], params1)
        registry.save_run(record1)
        gen1_artifact = record1.output.artifact

        # Generation 2: clip again
        out2 = workspace / "gen2.tif"
        params2 = ClipRasterParams(bounds=(-3.0, -3.0, 3.0, 3.0), output_path=str(out2))
        record2 = LocalExecutor().submit(ClipRasterOperator(), [gen1_artifact], params2)
        registry.save_run(record2)
        gen2_id = record2.output.artifact.id

        # Full lineage of gen2 should include gen1 and source
        full_lineage = registry.get_full_lineage(gen2_id)
        ancestor_ids = [row["artifact_id"] for row in full_lineage]
        assert gen1_artifact.id in ancestor_ids
        assert mat.artifact.id in ancestor_ids

    def test_no_lineage_for_source_artifact(self, sample_raster, workspace, registry):
        conn = LocalFileConnector()
        mat = conn.materialize(str(sample_raster), workspace)
        registry.save_artifact(mat.artifact)

        parents = registry.get_parents(mat.artifact.id)
        assert len(parents) == 0


class TestRegistryStats:
    """Registry can report its own state."""

    def test_stats(self, sample_raster, workspace, registry):
        conn = LocalFileConnector()
        mat = conn.materialize(str(sample_raster), workspace)
        registry.save_artifact(mat.artifact)

        output_path = workspace / "stats.tif"
        params = ClipRasterParams(
            bounds=(-5.0, -5.0, 5.0, 5.0),
            output_path=str(output_path),
        )
        record = LocalExecutor().submit(ClipRasterOperator(), [mat.artifact], params)
        registry.save_run(record)

        stats = registry.stats()
        assert stats["artifacts"] == 2  # source + output
        assert stats["runs"] == 1
        assert stats["checks"] > 0
        assert stats["lineage_edges"] == 1
