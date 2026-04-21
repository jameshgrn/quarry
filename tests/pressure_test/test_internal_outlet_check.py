"""Pressure test: InternalOutletCount check.

Exercises:
  - Standalone check on D8 flow direction artifacts
  - Operator-internal check via D8FlowDirectionOperator
  - Zero internal outlets on clean filled DEMs
  - Non-zero internal outlets when nodata holes create leaks
  - Check protocol compliance
  - Full chain integration (fill → D8 → check)

Failure signals:
  - Check misses cells flowing into nodata
  - False positives on clean DEMs
  - Standalone and operator-internal checks disagree
"""

from pathlib import Path

import numpy as np
import rasterio
from quarry_connectors.local_file import LocalFileConnector
from quarry_core.artifact import (
    Artifact,
    ArtifactType,
    BackingStore,
    BackingStoreKind,
    CheckResult,
    SpatialDescriptor,
    ValidationState,
)
from quarry_core.check import Check
from quarry_operators.checks import InternalOutletCount
from quarry_operators.d8_flow_direction import (
    NODATA,
    OUTLET,
    D8FlowDirectionOperator,
    D8FlowDirectionParams,
)
from quarry_operators.fill_depressions import FillDepressionsOperator, FillDepressionsParams
from rasterio.crs import CRS
from rasterio.transform import from_bounds

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_dem(path: Path, data: np.ndarray, nodata: float = -9999.0) -> Path:
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


def _write_flow(path: Path, flow: np.ndarray) -> Artifact:
    """Write a D8 flow direction raster and return its artifact."""
    h, w = flow.shape
    transform = from_bounds(0.0, 0.0, float(w), float(h), w, h)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=1,
        dtype="int16",
        crs=CRS.from_epsg(32618),
        transform=transform,
        nodata=NODATA,
    ) as dst:
        dst.write(flow.astype(np.int16), 1)

    return Artifact(
        type=ArtifactType.RASTER,
        name=path.stem,
        backing=BackingStore(
            kind=BackingStoreKind.LOCAL_FILE,
            uri=str(path),
        ),
        spatial=SpatialDescriptor(crs="EPSG:32618"),
    )


def _materialize(dem_path, workspace):
    conn = LocalFileConnector()
    return conn.materialize(str(dem_path), workspace).artifact


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


class TestProtocol:
    def test_satisfies_check_protocol(self):
        check = InternalOutletCount()
        assert isinstance(check, Check)

    def test_name(self):
        assert InternalOutletCount().name == "no_internal_outlets"

    def test_description_nonempty(self):
        assert len(InternalOutletCount().description) > 0

    def test_returns_check_result(self, tmp_path):
        # Simple valid flow grid — all outlets
        flow = np.full((3, 3), OUTLET, dtype=np.int16)
        art = _write_flow(tmp_path / "flow.tif", flow)
        result = InternalOutletCount().run(art)
        assert isinstance(result, CheckResult)


# ---------------------------------------------------------------------------
# Standalone check: clean grids
# ---------------------------------------------------------------------------


class TestCleanGrids:
    """Zero internal outlets on grids with no nodata holes."""

    def test_all_outlets(self, tmp_path):
        flow = np.full((5, 5), OUTLET, dtype=np.int16)
        art = _write_flow(tmp_path / "flow.tif", flow)
        result = InternalOutletCount().run(art)
        assert result.state == ValidationState.VALID

    def test_uniform_east_flow(self, tmp_path):
        """All cells flow east — rightmost column are outlets via boundary."""
        flow = np.full((5, 5), 0, dtype=np.int16)  # 0 = East
        # Right boundary cells need OUTLET code since they'd flow off-grid
        flow[:, -1] = OUTLET
        art = _write_flow(tmp_path / "flow.tif", flow)
        result = InternalOutletCount().run(art)
        assert result.state == ValidationState.VALID

    def test_uniform_south_flow(self, tmp_path):
        flow = np.full((5, 5), 2, dtype=np.int16)  # 2 = South
        flow[-1, :] = OUTLET
        art = _write_flow(tmp_path / "flow.tif", flow)
        result = InternalOutletCount().run(art)
        assert result.state == ValidationState.VALID

    def test_mixed_directions_no_nodata(self, tmp_path):
        """Grid with diverse directions, no nodata — zero internal outlets."""
        flow = np.array(
            [
                [OUTLET, OUTLET, OUTLET, OUTLET, OUTLET],
                [OUTLET, 2, 2, 2, OUTLET],
                [OUTLET, 0, 2, 4, OUTLET],
                [OUTLET, 0, 0, 4, OUTLET],
                [OUTLET, OUTLET, OUTLET, OUTLET, OUTLET],
            ],
            dtype=np.int16,
        )
        art = _write_flow(tmp_path / "flow.tif", flow)
        result = InternalOutletCount().run(art)
        assert result.state == ValidationState.VALID


# ---------------------------------------------------------------------------
# Standalone check: grids with nodata holes
# ---------------------------------------------------------------------------


class TestNodataHoles:
    """Non-zero internal outlets when flow directions point into nodata."""

    def test_single_cell_flows_into_nodata(self, tmp_path):
        """One cell flows east into a nodata cell."""
        flow = np.full((3, 3), OUTLET, dtype=np.int16)
        flow[1, 1] = NODATA  # nodata hole in center
        flow[1, 0] = 0  # flows east → into nodata
        art = _write_flow(tmp_path / "flow.tif", flow)
        result = InternalOutletCount().run(art)
        assert result.state == ValidationState.WARN
        assert "1 cells" in result.message

    def test_multiple_cells_flow_into_nodata(self, tmp_path):
        """Three cells flow into a nodata hole."""
        flow = np.full((5, 5), OUTLET, dtype=np.int16)
        flow[2, 2] = NODATA  # hole
        flow[2, 1] = 0  # east → nodata
        flow[1, 2] = 2  # south → nodata
        flow[2, 3] = 4  # west → nodata
        art = _write_flow(tmp_path / "flow.tif", flow)
        result = InternalOutletCount().run(art)
        assert result.state == ValidationState.WARN
        assert "3 cells" in result.message

    def test_nodata_row_creates_leaks(self, tmp_path):
        """Nodata row bisecting grid — cells flowing into it are internal outlets."""
        flow = np.full((5, 5), 2, dtype=np.int16)  # all flow south
        flow[-1, :] = OUTLET
        # Row 2 is nodata
        flow[2, :] = NODATA
        # Row 1 flows south → into nodata row
        art = _write_flow(tmp_path / "flow.tif", flow)
        result = InternalOutletCount().run(art)
        assert result.state == ValidationState.WARN
        # Row 1 has 5 cells flowing south into nodata row 2
        assert "5 cells" in result.message

    def test_nodata_at_boundary_no_leak(self, tmp_path):
        """Nodata at grid boundary — cells flowing off-grid skip the check."""
        flow = np.full((3, 3), OUTLET, dtype=np.int16)
        flow[0, :] = NODATA  # top row nodata
        # Remaining cells are all outlets — no one flows north into nodata
        art = _write_flow(tmp_path / "flow.tif", flow)
        result = InternalOutletCount().run(art)
        assert result.state == ValidationState.VALID


# ---------------------------------------------------------------------------
# Standalone check: edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_rejects_vector_artifact(self):
        art = Artifact(
            type=ArtifactType.VECTOR,
            name="not_a_raster",
            backing=BackingStore(kind=BackingStoreKind.LOCAL_FILE, uri="/fake.shp"),
        )
        result = InternalOutletCount().run(art)
        assert result.state == ValidationState.INVALID

    def test_rejects_missing_file(self):
        art = Artifact(
            type=ArtifactType.RASTER,
            name="missing",
            backing=BackingStore(kind=BackingStoreKind.LOCAL_FILE, uri="/nonexistent.tif"),
        )
        result = InternalOutletCount().run(art)
        assert result.state == ValidationState.INVALID

    def test_rejects_lazy_handle(self):
        art = Artifact(
            type=ArtifactType.RASTER,
            name="lazy",
            backing=BackingStore(kind=BackingStoreKind.LAZY_HANDLE, uri="/fake.tif"),
        )
        result = InternalOutletCount().run(art)
        assert result.state == ValidationState.INVALID

    def test_1x1_grid(self, tmp_path):
        flow = np.array([[OUTLET]], dtype=np.int16)
        art = _write_flow(tmp_path / "flow.tif", flow)
        result = InternalOutletCount().run(art)
        assert result.state == ValidationState.VALID

    def test_all_nodata(self, tmp_path):
        flow = np.full((3, 3), NODATA, dtype=np.int16)
        art = _write_flow(tmp_path / "flow.tif", flow)
        result = InternalOutletCount().run(art)
        assert result.state == ValidationState.VALID


# ---------------------------------------------------------------------------
# Operator-internal check: D8 emits no_internal_outlets
# ---------------------------------------------------------------------------


class TestOperatorIntegration:
    """D8FlowDirectionOperator now declares and runs no_internal_outlets."""

    def test_declared_in_operator(self):
        op = D8FlowDirectionOperator()
        assert "no_internal_outlets" in op.declared_checks()

    def test_clean_dem_passes(self, tmp_path):
        """Filled DEM → D8 should have zero internal outlets."""
        # Sloped DEM — no pits, no nodata
        rows = np.arange(5, 0, -1, dtype=np.float32)
        data = np.tile(rows[:, None], (1, 5))
        dem_path = _write_dem(tmp_path / "sloped.tif", data)
        art = _materialize(dem_path, tmp_path / "ws")

        op = D8FlowDirectionOperator()
        params = D8FlowDirectionParams(output_path=str(tmp_path / "d8.tif"))
        result = op.execute([art], params)

        outlet_checks = [c for c in result.checks if c.check_name == "no_internal_outlets"]
        assert len(outlet_checks) == 1
        assert outlet_checks[0].state == ValidationState.VALID

    def test_fill_then_d8_zero_internal_outlets(self, tmp_path):
        """Full fill → D8 chain: zero internal outlets guaranteed."""
        # Pit DEM
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
        dem_path = _write_dem(tmp_path / "pit.tif", data)
        art = _materialize(dem_path, tmp_path / "ws")

        # Fill
        fill_op = FillDepressionsOperator()
        fill_result = fill_op.execute(
            [art],
            FillDepressionsParams(output_path=str(tmp_path / "filled.tif")),
        )

        # D8
        d8_op = D8FlowDirectionOperator()
        d8_result = d8_op.execute(
            [fill_result.artifact],
            D8FlowDirectionParams(output_path=str(tmp_path / "d8.tif")),
        )

        outlet_checks = [c for c in d8_result.checks if c.check_name == "no_internal_outlets"]
        assert outlet_checks[0].state == ValidationState.VALID

    def test_nodata_dem_may_warn(self, tmp_path):
        """DEM with nodata hole — D8 may produce internal outlets (WARN)."""
        data = np.full((7, 7), 10.0, dtype=np.float32)
        nodata = -9999.0
        # Gentle slope south
        for r in range(7):
            data[r, :] += (6 - r) * 0.5
        # Nodata hole in center
        data[3, 3] = nodata

        dem_path = _write_dem(tmp_path / "nodata_hole.tif", data, nodata=nodata)
        art = _materialize(dem_path, tmp_path / "ws")

        # Fill then D8
        fill_result = FillDepressionsOperator().execute(
            [art],
            FillDepressionsParams(output_path=str(tmp_path / "filled.tif"), nodata=nodata),
        )
        d8_result = D8FlowDirectionOperator().execute(
            [fill_result.artifact],
            D8FlowDirectionParams(output_path=str(tmp_path / "d8.tif"), nodata=nodata),
        )

        outlet_checks = [c for c in d8_result.checks if c.check_name == "no_internal_outlets"]
        assert len(outlet_checks) == 1
        # With a nodata hole, cells around it may flow into it
        # The check catches this — state is WARN or VALID depending on geometry
        assert outlet_checks[0].state in (ValidationState.VALID, ValidationState.WARN)


# ---------------------------------------------------------------------------
# Standalone vs operator agreement
# ---------------------------------------------------------------------------


class TestStandaloneOperatorAgreement:
    """Standalone InternalOutletCount and operator-internal check agree."""

    def test_same_result_on_clean_dem(self, tmp_path):
        rows = np.arange(5, 0, -1, dtype=np.float32)
        data = np.tile(rows[:, None], (1, 5))
        dem_path = _write_dem(tmp_path / "sloped.tif", data)
        art = _materialize(dem_path, tmp_path / "ws")

        # Operator-internal
        op = D8FlowDirectionOperator()
        params = D8FlowDirectionParams(output_path=str(tmp_path / "d8.tif"))
        op_result = op.execute([art], params)
        op_check = next(c for c in op_result.checks if c.check_name == "no_internal_outlets")

        # Standalone
        standalone_check = InternalOutletCount().run(op_result.artifact)

        assert op_check.state == standalone_check.state

    def test_same_result_on_random_dem(self, tmp_path):
        rng = np.random.default_rng(99)
        data = rng.uniform(10.0, 100.0, size=(20, 20)).astype(np.float32)
        dem_path = _write_dem(tmp_path / "random.tif", data)
        art = _materialize(dem_path, tmp_path / "ws")

        # Fill → D8
        filled = FillDepressionsOperator().execute(
            [art],
            FillDepressionsParams(output_path=str(tmp_path / "filled.tif")),
        )
        d8_result = D8FlowDirectionOperator().execute(
            [filled.artifact],
            D8FlowDirectionParams(output_path=str(tmp_path / "d8.tif")),
        )
        op_check = next(c for c in d8_result.checks if c.check_name == "no_internal_outlets")

        standalone_check = InternalOutletCount().run(d8_result.artifact)

        assert op_check.state == standalone_check.state
