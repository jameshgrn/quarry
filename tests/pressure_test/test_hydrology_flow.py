"""Pressure test: end-to-end hydrology flow with registry integration.

Lane: flow

Exercises:
  - Full chain: fill_depressions → d8_flow_direction → flow_accumulation
  - Every artifact/run/check/lineage persisted to DuckDB registry
  - Lineage graph walkable (DEM → filled → D8 → accumulation)
  - Conservation invariant: outlet accumulation = total valid cells
  - Check propagation: all operator checks present in registry
  - Failure isolation: bad input stops chain at correct step
  - Registry round-trip: artifacts/runs survive save→load cycle

Failure signals:
  - Registry not capturing intermediate artifacts
  - Lineage edges missing between chain steps
  - Conservation broken through the full chain
  - Checks lost during registry persistence
  - Flow not stopping on operator failure
"""

from pathlib import Path

import numpy as np
import pytest
import rasterio
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
from quarry_operators.hydrology_flow import HydrologyFlow, HydrologyFlowParams
from quarry_registry.registry import Registry
from rasterio.crs import CRS
from rasterio.transform import from_bounds

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_dem(path: Path, data: np.ndarray, nodata: float = -9999.0) -> Path:
    """Write a single-band float32 DEM to a GeoTIFF."""
    h, w = data.shape
    transform = from_bounds(0.0, 0.0, float(w), float(h), w, h)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=1,
        dtype="float32",
        crs=CRS.from_epsg(32618),
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(data.astype(np.float32), 1)
    return path


@pytest.fixture
def pit_dem(tmp_path):
    """5x5 DEM with a single interior pit — canonical test surface."""
    data = np.array(
        [
            [9, 9, 9, 9, 9],
            [9, 7, 7, 7, 9],
            [9, 7, 3, 7, 9],
            [9, 7, 7, 7, 9],
            [9, 8, 8, 8, 9],
        ],
        dtype=np.float32,
    )
    return _make_dem(tmp_path / "pit_dem.tif", data)


@pytest.fixture
def sloped_dem(tmp_path):
    """10x10 DEM with uniform south slope — already drained, no pits."""
    rows = np.arange(10, 0, -1, dtype=np.float32)
    data = np.tile(rows[:, None], (1, 10))
    return _make_dem(tmp_path / "sloped_dem.tif", data)


@pytest.fixture
def random_dem(tmp_path):
    """30x30 random DEM — stresses fill + D8 + accumulation chain."""
    rng = np.random.default_rng(42)
    data = rng.uniform(10.0, 100.0, size=(30, 30)).astype(np.float32)
    return _make_dem(tmp_path / "random_dem.tif", data)


@pytest.fixture
def workspace(tmp_path):
    ws = tmp_path / "hydro_workspace"
    ws.mkdir()
    return ws


@pytest.fixture
def registry(tmp_path):
    return Registry(tmp_path / "registry_ws")


@pytest.fixture
def executor():
    return LocalExecutor()


def _materialize(dem_path, workspace):
    """Materialize a DEM through the connector."""
    conn = LocalFileConnector()
    return conn.materialize(str(dem_path), workspace).artifact


# ---------------------------------------------------------------------------
# Full chain tests
# ---------------------------------------------------------------------------


class TestHydrologyFlowEndToEnd:
    """The canonical hydrology chain executes and produces valid output."""

    def test_pit_dem_fills_and_drains(self, pit_dem, workspace, executor, registry):
        art = _materialize(pit_dem, workspace)
        flow = HydrologyFlow(executor=executor, registry=registry)
        result = flow.run(art, HydrologyFlowParams(workspace=workspace / "run"))

        assert result.success
        assert result.filled_dem is not None
        assert result.flow_direction is not None
        assert result.flow_accumulation is not None
        assert len(result.runs) == 3
        assert all(r.status == RunStatus.COMPLETED for r in result.runs)

    def test_sloped_dem_passthrough(self, sloped_dem, workspace, executor, registry):
        """Already-drained DEM passes through fill unchanged."""
        art = _materialize(sloped_dem, workspace)
        flow = HydrologyFlow(executor=executor, registry=registry)
        result = flow.run(art, HydrologyFlowParams(workspace=workspace / "run"))

        assert result.success
        assert len(result.artifacts) == 3

    def test_random_dem_full_chain(self, random_dem, workspace, executor, registry):
        """Random DEM exercises all three operators under realistic conditions."""
        art = _materialize(random_dem, workspace)
        flow = HydrologyFlow(executor=executor, registry=registry)
        result = flow.run(art, HydrologyFlowParams(workspace=workspace / "run"))

        assert result.success
        assert len(result.runs) == 3

    def test_all_output_files_exist(self, pit_dem, workspace, executor):
        art = _materialize(pit_dem, workspace)
        flow = HydrologyFlow(executor=executor)
        run_dir = workspace / "run"
        result = flow.run(art, HydrologyFlowParams(workspace=run_dir))

        assert result.success
        assert (run_dir / "filled_dem.tif").exists()
        assert (run_dir / "flow_direction.tif").exists()
        assert (run_dir / "flow_accumulation.tif").exists()


# ---------------------------------------------------------------------------
# Conservation invariant
# ---------------------------------------------------------------------------


class TestConservation:
    """Flow accumulation conservation: sum at outlets = total valid cells × weight."""

    def test_conservation_pit_dem(self, pit_dem, workspace, executor):
        art = _materialize(pit_dem, workspace)
        flow = HydrologyFlow(executor=executor)
        result = flow.run(art, HydrologyFlowParams(workspace=workspace / "run"))

        assert result.success
        conservation_checks = [c for c in result.all_checks if c.check_name == "conservation"]
        assert len(conservation_checks) == 1
        assert conservation_checks[0].state == ValidationState.VALID

    def test_conservation_random_dem(self, random_dem, workspace, executor):
        art = _materialize(random_dem, workspace)
        flow = HydrologyFlow(executor=executor)
        result = flow.run(art, HydrologyFlowParams(workspace=workspace / "run"))

        assert result.success
        conservation_checks = [c for c in result.all_checks if c.check_name == "conservation"]
        assert len(conservation_checks) == 1
        assert conservation_checks[0].state == ValidationState.VALID

    def test_conservation_with_custom_weight(self, pit_dem, workspace, executor):
        art = _materialize(pit_dem, workspace)
        flow = HydrologyFlow(executor=executor)
        result = flow.run(art, HydrologyFlowParams(workspace=workspace / "run", weight=2.5))

        assert result.success
        # Read the accumulation raster and verify conservation manually
        acc_path = workspace / "run" / "flow_accumulation.tif"
        with rasterio.open(acc_path) as src:
            acc = src.read(1)
        # With weight=2.5 and 25 valid cells, total should be 62.5
        # Outlets are boundary cells that collect everything
        fdir_path = workspace / "run" / "flow_direction.tif"
        with rasterio.open(fdir_path) as src:
            fdir = src.read(1)
        # Outlet cells have code 8
        outlet_mask = fdir == 8
        outlet_sum = float(acc[outlet_mask].sum())
        total_valid = int(np.sum((fdir >= 0) & (fdir <= 9)))
        assert abs(outlet_sum - total_valid * 2.5) < 1e-6


# ---------------------------------------------------------------------------
# Check propagation
# ---------------------------------------------------------------------------


class TestCheckPropagation:
    """All operator checks are present in the flow result and registry."""

    def test_all_checks_collected(self, pit_dem, workspace, executor):
        art = _materialize(pit_dem, workspace)
        flow = HydrologyFlow(executor=executor)
        result = flow.run(art, HydrologyFlowParams(workspace=workspace / "run"))

        assert result.success
        check_names = {c.check_name for c in result.all_checks}
        # FillDepressions checks
        assert "no_interior_pits" in check_names
        assert "elevation_only_raised" in check_names
        # D8 checks
        assert "valid_code_set" in check_names
        assert "no_pits" in check_names
        assert "all_valid_assigned" in check_names
        # FlowAccumulation checks
        assert "no_cycles" in check_names
        assert "nonnegative" in check_names
        assert "conservation" in check_names

    def test_no_invalid_checks_on_valid_input(self, pit_dem, workspace, executor):
        art = _materialize(pit_dem, workspace)
        flow = HydrologyFlow(executor=executor)
        result = flow.run(art, HydrologyFlowParams(workspace=workspace / "run"))

        assert result.success
        assert not result.has_invalid_checks

    def test_checks_persisted_to_registry(self, pit_dem, workspace, executor, registry):
        art = _materialize(pit_dem, workspace)
        flow = HydrologyFlow(executor=executor, registry=registry)
        result = flow.run(art, HydrologyFlowParams(workspace=workspace / "run"))

        assert result.success
        # Each output artifact should have checks in registry
        for artifact in result.artifacts:
            checks = registry.get_checks(artifact_id=artifact.id)
            assert len(checks) > 0, f"No checks for artifact {artifact.name}"

    def test_checks_linked_to_runs(self, pit_dem, workspace, executor, registry):
        art = _materialize(pit_dem, workspace)
        flow = HydrologyFlow(executor=executor, registry=registry)
        result = flow.run(art, HydrologyFlowParams(workspace=workspace / "run"))

        assert result.success
        for run_record in result.runs:
            checks = registry.get_checks(run_id=run_record.id)
            assert len(checks) > 0, f"No checks for run {run_record.operator_name}"


# ---------------------------------------------------------------------------
# Registry persistence
# ---------------------------------------------------------------------------


class TestRegistryPersistence:
    """All artifacts, runs, and lineage edges persist correctly."""

    def test_all_artifacts_in_registry(self, pit_dem, workspace, executor, registry):
        art = _materialize(pit_dem, workspace)
        flow = HydrologyFlow(executor=executor, registry=registry)
        result = flow.run(art, HydrologyFlowParams(workspace=workspace / "run"))

        assert result.success
        # Input + 3 outputs = 4 artifacts
        stats = registry.stats()
        assert stats["artifacts"] == 4

    def test_all_runs_in_registry(self, pit_dem, workspace, executor, registry):
        art = _materialize(pit_dem, workspace)
        flow = HydrologyFlow(executor=executor, registry=registry)
        result = flow.run(art, HydrologyFlowParams(workspace=workspace / "run"))

        assert result.success
        stats = registry.stats()
        assert stats["runs"] == 3

    def test_runs_retrievable_by_id(self, pit_dem, workspace, executor, registry):
        art = _materialize(pit_dem, workspace)
        flow = HydrologyFlow(executor=executor, registry=registry)
        result = flow.run(art, HydrologyFlowParams(workspace=workspace / "run"))

        assert result.success
        for run_record in result.runs:
            loaded = registry.get_run(run_record.id)
            assert loaded is not None
            assert loaded.operator_name == run_record.operator_name
            assert loaded.status == RunStatus.COMPLETED

    def test_artifacts_round_trip(self, pit_dem, workspace, executor, registry):
        art = _materialize(pit_dem, workspace)
        flow = HydrologyFlow(executor=executor, registry=registry)
        result = flow.run(art, HydrologyFlowParams(workspace=workspace / "run"))

        assert result.success
        for artifact in result.artifacts:
            loaded = registry.get_artifact(artifact.id)
            assert loaded is not None
            assert loaded.type == ArtifactType.RASTER
            assert loaded.spatial.crs is not None
            assert loaded.backing is not None


# ---------------------------------------------------------------------------
# Lineage graph
# ---------------------------------------------------------------------------


class TestLineageGraph:
    """Lineage edges form a connected chain through the registry."""

    def test_lineage_edges_exist(self, pit_dem, workspace, executor, registry):
        art = _materialize(pit_dem, workspace)
        flow = HydrologyFlow(executor=executor, registry=registry)
        result = flow.run(art, HydrologyFlowParams(workspace=workspace / "run"))

        assert result.success
        stats = registry.stats()
        # 3 edges: DEM→filled, filled→D8, D8→accumulation
        assert stats["lineage_edges"] == 3

    def test_parent_child_relationships(self, pit_dem, workspace, executor, registry):
        art = _materialize(pit_dem, workspace)
        flow = HydrologyFlow(executor=executor, registry=registry)
        result = flow.run(art, HydrologyFlowParams(workspace=workspace / "run"))

        assert result.success
        # filled_dem parent = input DEM
        parents = registry.get_parents(result.filled_dem.id)
        assert len(parents) == 1
        assert parents[0]["artifact_id"] == art.id
        assert parents[0]["operation"] == "fill_depressions"

        # flow_direction parent = filled_dem
        parents = registry.get_parents(result.flow_direction.id)
        assert len(parents) == 1
        assert parents[0]["artifact_id"] == result.filled_dem.id
        assert parents[0]["operation"] == "d8_flow_direction"

        # flow_accumulation parent = flow_direction
        parents = registry.get_parents(result.flow_accumulation.id)
        assert len(parents) == 1
        assert parents[0]["artifact_id"] == result.flow_direction.id
        assert parents[0]["operation"] == "flow_accumulation"

    def test_full_ancestor_chain(self, pit_dem, workspace, executor, registry):
        """Walking ancestors from accumulation reaches the input DEM."""
        art = _materialize(pit_dem, workspace)
        flow = HydrologyFlow(executor=executor, registry=registry)
        result = flow.run(art, HydrologyFlowParams(workspace=workspace / "run"))

        assert result.success
        lineage = registry.get_full_lineage(result.flow_accumulation.id)
        ancestor_ids = {entry["artifact_id"] for entry in lineage}
        # Should contain flow_direction, filled_dem, and input DEM
        assert result.flow_direction.id in ancestor_ids
        assert result.filled_dem.id in ancestor_ids
        assert art.id in ancestor_ids

    def test_children_from_input(self, pit_dem, workspace, executor, registry):
        art = _materialize(pit_dem, workspace)
        flow = HydrologyFlow(executor=executor, registry=registry)
        result = flow.run(art, HydrologyFlowParams(workspace=workspace / "run"))

        assert result.success
        children = registry.get_children(art.id)
        assert len(children) == 1
        assert children[0]["artifact_id"] == result.filled_dem.id


# ---------------------------------------------------------------------------
# Failure isolation
# ---------------------------------------------------------------------------


class TestFailureIsolation:
    """Flow stops at the correct step when given bad input."""

    def test_bad_input_type_stops_at_fill(self, workspace, executor):
        """Vector artifact rejected at fill_depressions step."""
        bad_art = Artifact(
            type=ArtifactType.VECTOR,
            name="not_a_dem",
            backing=BackingStore(kind=BackingStoreKind.LOCAL_FILE, uri="/fake.shp"),
        )
        flow = HydrologyFlow(executor=executor)
        result = flow.run(bad_art, HydrologyFlowParams(workspace=workspace / "run"))

        assert not result.success
        assert result.failed_step == "fill_depressions"
        assert result.filled_dem is None
        assert result.flow_direction is None
        assert result.flow_accumulation is None

    def test_lazy_input_stops_at_fill(self, workspace, executor):
        """Lazy artifact rejected (not materialized)."""
        lazy_art = Artifact(
            type=ArtifactType.RASTER,
            name="lazy_dem",
            backing=BackingStore(kind=BackingStoreKind.LAZY_HANDLE, uri="/fake.tif"),
        )
        flow = HydrologyFlow(executor=executor)
        result = flow.run(lazy_art, HydrologyFlowParams(workspace=workspace / "run"))

        assert not result.success
        assert result.failed_step == "fill_depressions"

    def test_failed_flow_persists_nothing_extra(self, workspace, executor, registry):
        """Failed flow persists the attempted run, but no output artifacts."""
        bad_art = Artifact(
            type=ArtifactType.VECTOR,
            name="not_a_dem",
            backing=BackingStore(kind=BackingStoreKind.LOCAL_FILE, uri="/fake.shp"),
        )
        flow = HydrologyFlow(executor=executor, registry=registry)
        result = flow.run(bad_art, HydrologyFlowParams(workspace=workspace / "run"))

        assert not result.success
        stats = registry.stats()
        # Only the input artifact was saved, but the failed run is recorded
        assert stats["artifacts"] == 1
        assert stats["runs"] == 1


# ---------------------------------------------------------------------------
# Without registry (pure execution)
# ---------------------------------------------------------------------------


class TestNoRegistry:
    """Flow works without a registry — pure execution mode."""

    def test_executes_without_registry(self, pit_dem, workspace, executor):
        art = _materialize(pit_dem, workspace)
        flow = HydrologyFlow(executor=executor)
        result = flow.run(art, HydrologyFlowParams(workspace=workspace / "run"))

        assert result.success
        assert len(result.runs) == 3
        assert result.flow_accumulation is not None

    def test_checks_still_collected(self, pit_dem, workspace, executor):
        art = _materialize(pit_dem, workspace)
        flow = HydrologyFlow(executor=executor)
        result = flow.run(art, HydrologyFlowParams(workspace=workspace / "run"))

        assert result.success
        assert len(result.all_checks) > 0


# ---------------------------------------------------------------------------
# Regression fixtures
# ---------------------------------------------------------------------------


class TestRegressionFixtures:
    """Fixed synthetic surfaces that must always produce known outcomes."""

    def test_flat_surface_drains_to_boundary(self, tmp_path, executor):
        """Completely flat DEM — all cells should drain to boundary."""
        data = np.full((5, 5), 10.0, dtype=np.float32)
        dem_path = _make_dem(tmp_path / "flat.tif", data)
        art = _materialize(dem_path, tmp_path / "ws")
        flow = HydrologyFlow(executor=executor)
        result = flow.run(art, HydrologyFlowParams(workspace=tmp_path / "run"))

        assert result.success
        # Conservation must hold even on flat surface
        conservation = [c for c in result.all_checks if c.check_name == "conservation"]
        assert conservation[0].state == ValidationState.VALID

    def test_v_shaped_valley(self, tmp_path, executor):
        """V-shaped valley — water flows to center channel then south."""
        data = np.zeros((10, 10), dtype=np.float32)
        for r in range(10):
            for c in range(10):
                # Distance from center column
                dist = abs(c - 4.5)
                # Slope south
                south_slope = (9 - r) * 0.1
                data[r, c] = dist + south_slope + 5.0
        dem_path = _make_dem(tmp_path / "valley.tif", data)
        art = _materialize(dem_path, tmp_path / "ws")
        flow = HydrologyFlow(executor=executor)
        result = flow.run(art, HydrologyFlowParams(workspace=tmp_path / "run"))

        assert result.success
        # The bottom-center cells should have highest accumulation
        acc_path = tmp_path / "run" / "flow_accumulation.tif"
        with rasterio.open(acc_path) as src:
            acc = src.read(1)
        # Bottom row center columns should be among the highest values
        bottom_center = acc[9, 4:6].max()
        assert bottom_center > acc.mean()

    def test_nodata_ring(self, tmp_path, executor):
        """DEM with nodata ring — interior should still drain correctly."""
        data = np.full((10, 10), 50.0, dtype=np.float32)
        nodata = -9999.0
        # Create nodata ring at row/col 2
        data[2, :] = nodata
        data[:, 2] = nodata
        # Interior (rows 0-1, cols 0-1) is a small 2x2 block with slope
        data[0, 0] = 10.0
        data[0, 1] = 9.0
        data[1, 0] = 8.0
        data[1, 1] = 7.0
        # Exterior has gentle slope
        for r in range(3, 10):
            for c in range(3, 10):
                data[r, c] = 50.0 - r * 0.5

        dem_path = _make_dem(tmp_path / "nodata_ring.tif", data, nodata=nodata)
        art = _materialize(dem_path, tmp_path / "ws")
        flow = HydrologyFlow(executor=executor)
        result = flow.run(art, HydrologyFlowParams(workspace=tmp_path / "run"))

        assert result.success
        # Conservation check should still pass (nodata excluded)
        conservation = [c for c in result.all_checks if c.check_name == "conservation"]
        assert conservation[0].state == ValidationState.VALID

    def test_steep_cone(self, tmp_path, executor):
        """Steep cone — radial outward flow, single peak at center."""
        data = np.zeros((11, 11), dtype=np.float32)
        for r in range(11):
            for c in range(11):
                data[r, c] = 100.0 - np.sqrt((r - 5) ** 2 + (c - 5) ** 2) * 10
        # Ensure boundary is lowest
        data = np.maximum(data, 0.0)
        dem_path = _make_dem(tmp_path / "cone.tif", data)
        art = _materialize(dem_path, tmp_path / "ws")
        flow = HydrologyFlow(executor=executor)
        result = flow.run(art, HydrologyFlowParams(workspace=tmp_path / "run"))

        assert result.success
        # Center cell should have accumulation = 1 (only itself, it's the peak)
        acc_path = tmp_path / "run" / "flow_accumulation.tif"
        with rasterio.open(acc_path) as src:
            acc = src.read(1)
        assert acc[5, 5] == pytest.approx(1.0)

    def test_nested_pits(self, tmp_path, executor):
        """Two nested depressions — both must fill before flow routes."""
        data = np.full((7, 7), 20.0, dtype=np.float32)
        # Outer depression
        data[1:6, 1:6] = 10.0
        # Inner depression
        data[2:5, 2:5] = 5.0
        data[3, 3] = 1.0  # deepest point
        # Spill point (one boundary cell lower)
        data[0, 3] = 8.0
        dem_path = _make_dem(tmp_path / "nested.tif", data)
        art = _materialize(dem_path, tmp_path / "ws")
        flow = HydrologyFlow(executor=executor)
        result = flow.run(art, HydrologyFlowParams(workspace=tmp_path / "run"))

        assert result.success
        # No pits should remain after fill
        pit_checks = [c for c in result.all_checks if c.check_name == "no_interior_pits"]
        assert pit_checks[0].state == ValidationState.VALID
        # Conservation still holds
        conservation = [c for c in result.all_checks if c.check_name == "conservation"]
        assert conservation[0].state == ValidationState.VALID
