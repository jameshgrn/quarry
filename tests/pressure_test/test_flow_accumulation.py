"""FlowAccumulation operator pressure test.

Stress points:
1. Simple linear chain: accumulation increases downstream
2. Branching: confluence cells sum upstream contributions
3. OUTLET cells accumulate all upstream flow (conservation)
4. PIT cells retain only self-weight (no downstream propagation)
5. Nodata cells excluded from accumulation
6. Cycle detection rejects invalid flow grids
7. Conservation check: total outlet accumulation = total input weight
8. Nonnegative: every valid cell >= weight
9. Full chain: fill → D8 → accumulation on random DEM
10. Operator protocol compliance
"""

import numpy as np
import pytest
import rasterio
from quarry_core.artifact import (
    Artifact,
    ArtifactType,
    BackingStore,
    BackingStoreKind,
    ValidationState,
)
from quarry_core.operator import Operator, OperatorError, ResourceScale
from quarry_operators.flow_accumulation import (
    NODATA_CODE,
    OUTLET,
    PIT,
    FlowAccumulationOperator,
    FlowAccumulationParams,
    _accumulate_toposort,
    _detect_cycle,
)
from rasterio.crs import CRS
from rasterio.transform import from_bounds

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_flow(path, data):
    """Write a single-band flow direction GeoTIFF (int16)."""
    nrows, ncols = data.shape
    transform = from_bounds(0, 0, ncols, nrows, ncols, nrows)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=nrows,
        width=ncols,
        count=1,
        dtype="int16",
        crs=CRS.from_epsg(32610),
        transform=transform,
        nodata=NODATA_CODE,
    ) as dst:
        dst.write(data.astype(np.int16), 1)


def _make_artifact(path):
    """Create an Artifact for a local raster file."""
    from quarry_core.artifact import SpatialDescriptor, content_hash

    with rasterio.open(path) as src:
        bounds = src.bounds
        return Artifact(
            type=ArtifactType.RASTER,
            name=path.stem,
            backing=BackingStore(
                kind=BackingStoreKind.LOCAL_FILE,
                uri=str(path),
                size_bytes=path.stat().st_size,
                content_hash=content_hash(path),
            ),
            spatial=SpatialDescriptor(
                crs=str(src.crs),
                extent=(bounds.left, bounds.bottom, bounds.right, bounds.top),
                resolution=(src.res[0], src.res[1]),
                band_count=src.count,
            ),
        )


@pytest.fixture
def op():
    return FlowAccumulationOperator()


@pytest.fixture
def linear_flow(tmp_path):
    """5x1 linear chain flowing east, last cell is OUTLET.

    [0] → [0] → [0] → [0] → [OUTLET]
    Expected accumulation: 1, 2, 3, 4, 5
    """
    flow = np.array([[0, 0, 0, 0, OUTLET]], dtype=np.int8)
    path = tmp_path / "linear.tif"
    _write_flow(path, flow)
    return path, flow


@pytest.fixture
def branching_flow(tmp_path):
    """3x3 grid: top row flows south, middle row flows east, converges at (1,2).

    Flow codes: 0=E, 2=S, 8=OUTLET
      [2] [2] [2]
      [0] [0] [OUTLET]
      [6] [6] [6]     (6=N, flowing north into middle row)

    Middle-right (1,2) is the outlet collecting everything.
    """
    flow = np.array(
        [
            [2, 2, 2],
            [0, 0, OUTLET],
            [6, 6, 6],
        ],
        dtype=np.int8,
    )
    path = tmp_path / "branching.tif"
    _write_flow(path, flow)
    return path, flow


@pytest.fixture
def nodata_flow(tmp_path):
    """Flow grid with nodata hole."""
    flow = np.array(
        [
            [0, 0, OUTLET],
            [0, NODATA_CODE, 0],
            [0, 0, OUTLET],
        ],
        dtype=np.int8,
    )
    path = tmp_path / "nodata_flow.tif"
    _write_flow(path, flow)
    return path, flow


@pytest.fixture
def pit_flow(tmp_path):
    """Flow grid with a PIT cell (isolated, receives no upstream)."""
    flow = np.array(
        [
            [0, 0, OUTLET],
            [PIT, 0, OUTLET],
            [0, 0, OUTLET],
        ],
        dtype=np.int8,
    )
    path = tmp_path / "pit_flow.tif"
    _write_flow(path, flow)
    return path, flow


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


class TestProtocolCompliance:
    def test_implements_operator_protocol(self, op):
        assert isinstance(op, Operator)

    def test_name(self, op):
        assert op.name == "flow_accumulation"

    def test_spec_shape(self, op):
        spec = op.spec
        assert spec.input_types == (ArtifactType.RASTER,)
        assert spec.output_type == ArtifactType.RASTER
        assert spec.min_inputs == 1
        assert spec.max_inputs == 1
        assert spec.resource_scale == ResourceScale.MEDIUM

    def test_declared_checks(self, op):
        checks = op.declared_checks()
        assert "no_cycles" in checks
        assert "nonnegative" in checks
        assert "conservation" in checks
        assert "backing_accessible" in checks


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_rejects_no_inputs(self, op):
        errors = op.validate_inputs([], FlowAccumulationParams(output_path="/tmp/x.tif"))
        assert any("required" in e.lower() for e in errors)

    def test_rejects_vector_input(self, op):
        art = Artifact(
            type=ArtifactType.VECTOR,
            name="v",
            backing=BackingStore(
                kind=BackingStoreKind.LOCAL_FILE, uri="/fake", size_bytes=0, content_hash=""
            ),
        )
        errors = op.validate_inputs([art], FlowAccumulationParams(output_path="/tmp/x.tif"))
        assert any("raster" in e.lower() for e in errors)

    def test_rejects_missing_output_path(self, op, linear_flow):
        path, _ = linear_flow
        art = _make_artifact(path)
        errors = op.validate_inputs([art], FlowAccumulationParams(output_path=""))
        assert any("output_path" in e for e in errors)

    def test_rejects_bad_weight(self, op, linear_flow):
        path, _ = linear_flow
        art = _make_artifact(path)
        errors = op.validate_inputs(
            [art], FlowAccumulationParams(output_path="/tmp/x.tif", weight=-1.0)
        )
        assert any("weight" in e for e in errors)

    def test_accepts_valid_input(self, op, linear_flow):
        path, _ = linear_flow
        art = _make_artifact(path)
        errors = op.validate_inputs([art], FlowAccumulationParams(output_path="/tmp/x.tif"))
        assert errors == []


# ---------------------------------------------------------------------------
# Algorithm correctness
# ---------------------------------------------------------------------------


class TestAccumulation:
    def test_linear_chain(self, linear_flow):
        """Linear chain: accumulation = position from headwater."""
        _, flow = linear_flow
        valid = (flow >= 0) & (flow <= PIT)
        acc = _accumulate_toposort(flow, valid, weight=1.0)
        # 5 cells flowing east: [1, 2, 3, 4, 5]
        np.testing.assert_array_almost_equal(acc[0], [1.0, 2.0, 3.0, 4.0, 5.0])

    def test_branching_confluence(self, branching_flow):
        """Branching: outlet accumulates all upstream cells."""
        _, flow = branching_flow
        valid = (flow >= 0) & (flow <= PIT)
        acc = _accumulate_toposort(flow, valid, weight=1.0)
        # All 9 cells drain to (1,2) — outlet should have acc=9
        assert acc[1, 2] == 9.0

    def test_branching_intermediate(self, branching_flow):
        """Middle cells accumulate contributions from above and below."""
        _, flow = branching_flow
        valid = (flow >= 0) & (flow <= PIT)
        acc = _accumulate_toposort(flow, valid, weight=1.0)
        # (1,0) gets flow from (0,0) and (2,0) plus self = 3
        assert acc[1, 0] == 3.0
        # (1,1) gets (1,0)=3 + (0,1)=1 + (2,1)=1 + self=1 = 6
        assert acc[1, 1] == 6.0

    def test_nodata_excluded(self, nodata_flow):
        """Nodata cells don't contribute to accumulation."""
        _, flow = nodata_flow
        valid = (flow >= 0) & (flow <= PIT)
        acc = _accumulate_toposort(flow, valid, weight=1.0)
        # Nodata cell should have 0 accumulation (excluded)
        assert acc[1, 1] == 0.0

    def test_pit_retains_self(self, pit_flow):
        """PIT cells accumulate self-weight but don't pass downstream."""
        _, flow = pit_flow
        valid = (flow >= 0) & (flow <= PIT)
        acc = _accumulate_toposort(flow, valid, weight=1.0)
        # PIT at (1,0): acc = 1 (self only, no upstream, no downstream pass)
        assert acc[1, 0] == 1.0

    def test_custom_weight(self, linear_flow):
        """Custom weight scales accumulation."""
        _, flow = linear_flow
        valid = (flow >= 0) & (flow <= PIT)
        acc = _accumulate_toposort(flow, valid, weight=2.5)
        np.testing.assert_array_almost_equal(acc[0], [2.5, 5.0, 7.5, 10.0, 12.5])

    def test_conservation_property(self, branching_flow):
        """Total outlet accumulation = total input weight (conservation)."""
        _, flow = branching_flow
        valid = (flow >= 0) & (flow <= PIT)
        acc = _accumulate_toposort(flow, valid, weight=1.0)
        total_weight = float(np.sum(valid)) * 1.0
        outlet_mask = valid & ((flow == OUTLET) | (flow == PIT))
        outlet_sum = float(np.sum(acc[outlet_mask]))
        assert abs(outlet_sum - total_weight) < 1e-10


# ---------------------------------------------------------------------------
# Cycle detection
# ---------------------------------------------------------------------------


class TestCycleDetection:
    def test_no_cycle_in_valid_flow(self, linear_flow):
        _, flow = linear_flow
        valid = (flow >= 0) & (flow <= PIT)
        assert not _detect_cycle(flow, valid)

    def test_detects_two_cell_cycle(self):
        """Two cells pointing at each other = cycle."""
        # (0,0) flows east (code 0), (0,1) flows west (code 4)
        flow = np.array([[0, 4]], dtype=np.int8)
        valid = np.ones(flow.shape, dtype=bool)
        assert _detect_cycle(flow, valid)

    def test_detects_ring_cycle(self):
        """3x3 ring of cells flowing in a circle."""
        # Create a cycle: (0,0)→E, (0,1)→S, (1,1)→W, (1,0)→N
        flow = np.array(
            [
                [0, 2],
                [6, 4],
            ],
            dtype=np.int8,
        )
        valid = np.ones(flow.shape, dtype=bool)
        assert _detect_cycle(flow, valid)

    def test_no_cycle_with_outlets(self, branching_flow):
        _, flow = branching_flow
        valid = (flow >= 0) & (flow <= PIT)
        assert not _detect_cycle(flow, valid)


# ---------------------------------------------------------------------------
# Full operator execution
# ---------------------------------------------------------------------------


class TestExecution:
    def test_execute_produces_raster(self, op, linear_flow, tmp_path):
        path, _ = linear_flow
        art = _make_artifact(path)
        output = tmp_path / "acc.tif"
        params = FlowAccumulationParams(output_path=str(output))

        result = op.execute([art], params)
        assert result.artifact.type == ArtifactType.RASTER
        assert output.exists()

    def test_execute_checks_all_pass(self, op, branching_flow, tmp_path):
        path, _ = branching_flow
        art = _make_artifact(path)
        output = tmp_path / "acc.tif"
        params = FlowAccumulationParams(output_path=str(output))

        result = op.execute([art], params)
        for check in result.checks:
            assert check.state == ValidationState.VALID, f"{check.check_name}: {check.message}"

    def test_execute_rejects_cycle(self, op, tmp_path):
        """Cyclic flow grid should raise OperatorError."""
        flow = np.array([[0, 4]], dtype=np.int8)
        path = tmp_path / "cycle.tif"
        _write_flow(path, flow)
        art = _make_artifact(path)
        output = tmp_path / "acc.tif"
        params = FlowAccumulationParams(output_path=str(output))

        with pytest.raises(OperatorError, match="[Cc]ycle"):
            op.execute([art], params)

    def test_execute_lineage(self, op, linear_flow, tmp_path):
        path, _ = linear_flow
        art = _make_artifact(path)
        output = tmp_path / "acc.tif"
        params = FlowAccumulationParams(output_path=str(output))

        result = op.execute([art], params)
        assert result.artifact.lineage.operation == "flow_accumulation"
        assert art.id in result.artifact.lineage.inputs
        assert result.artifact.lineage.params["weight"] == 1.0

    def test_execute_metadata(self, op, linear_flow, tmp_path):
        path, _ = linear_flow
        art = _make_artifact(path)
        output = tmp_path / "acc.tif"
        params = FlowAccumulationParams(output_path=str(output))

        result = op.execute([art], params)
        assert result.artifact.metadata["algorithm"] == "topological_sort_kahn"


# ---------------------------------------------------------------------------
# Integration: full hydrology chain
# ---------------------------------------------------------------------------


class TestFullChain:
    def test_fill_d8_accumulation_chain(self, tmp_path):
        """Full chain: FillDepressions → D8 → FlowAccumulation on random DEM."""
        from quarry_operators.d8_flow_direction import (
            D8FlowDirectionOperator,
            D8FlowDirectionParams,
        )
        from quarry_operators.fill_depressions import (
            FillDepressionsOperator,
            FillDepressionsParams,
        )

        # Random 30x30 DEM
        rng = np.random.default_rng(99)
        dem = rng.uniform(0, 100, size=(30, 30)).astype(np.float64)

        def _write_dem(path, data):
            nrows, ncols = data.shape
            transform = from_bounds(0, 0, ncols, nrows, ncols, nrows)
            with rasterio.open(
                path,
                "w",
                driver="GTiff",
                height=nrows,
                width=ncols,
                count=1,
                dtype="float64",
                crs=CRS.from_epsg(32610),
                transform=transform,
            ) as dst:
                dst.write(data, 1)

        input_path = tmp_path / "dem.tif"
        _write_dem(input_path, dem)
        raw_art = _make_artifact(input_path)

        # Step 1: Fill
        fill_op = FillDepressionsOperator()
        filled_path = tmp_path / "filled.tif"
        fill_result = fill_op.execute(
            [raw_art],
            FillDepressionsParams(output_path=str(filled_path), apply_gradient=True),
        )

        # Step 2: D8
        d8_op = D8FlowDirectionOperator()
        flow_path = tmp_path / "flow.tif"
        d8_result = d8_op.execute(
            [fill_result.artifact],
            D8FlowDirectionParams(output_path=str(flow_path)),
        )

        # Step 3: Accumulation
        acc_op = FlowAccumulationOperator()
        acc_path = tmp_path / "acc.tif"
        acc_result = acc_op.execute(
            [d8_result.artifact],
            FlowAccumulationParams(output_path=str(acc_path)),
        )

        # All checks pass
        for check in acc_result.checks:
            assert check.state == ValidationState.VALID, f"{check.check_name}: {check.message}"

        # Accumulation values make sense
        with rasterio.open(acc_path) as src:
            acc = src.read(1)

        valid = acc >= 0
        assert np.min(acc[valid]) >= 1.0  # minimum is self-weight
        assert np.max(acc[valid]) <= float(np.sum(valid))  # max can't exceed total cells

    def test_chain_conservation(self, tmp_path):
        """Conservation: sum at outlets equals total valid cell count."""
        from quarry_operators.d8_flow_direction import (
            D8FlowDirectionOperator,
            D8FlowDirectionParams,
        )
        from quarry_operators.fill_depressions import (
            FillDepressionsOperator,
            FillDepressionsParams,
        )

        # Simple sloped DEM (no depressions)
        dem = np.zeros((10, 10), dtype=np.float64)
        for r in range(10):
            dem[r, :] = 9.0 - r  # slope south

        def _write_dem(path, data):
            nrows, ncols = data.shape
            transform = from_bounds(0, 0, ncols, nrows, ncols, nrows)
            with rasterio.open(
                path,
                "w",
                driver="GTiff",
                height=nrows,
                width=ncols,
                count=1,
                dtype="float64",
                crs=CRS.from_epsg(32610),
                transform=transform,
            ) as dst:
                dst.write(data, 1)

        input_path = tmp_path / "slope.tif"
        _write_dem(input_path, dem)
        raw_art = _make_artifact(input_path)

        # Fill (no-op on sloped DEM)
        fill_op = FillDepressionsOperator()
        filled_path = tmp_path / "filled.tif"
        fill_result = fill_op.execute(
            [raw_art],
            FillDepressionsParams(output_path=str(filled_path), apply_gradient=False),
        )

        # D8
        d8_op = D8FlowDirectionOperator()
        flow_path = tmp_path / "flow.tif"
        d8_result = d8_op.execute(
            [fill_result.artifact],
            D8FlowDirectionParams(output_path=str(flow_path)),
        )

        # Accumulation
        acc_op = FlowAccumulationOperator()
        acc_path = tmp_path / "acc.tif"
        acc_result = acc_op.execute(
            [d8_result.artifact],
            FlowAccumulationParams(output_path=str(acc_path)),
        )

        # Conservation check should pass
        cons_check = next(c for c in acc_result.checks if c.check_name == "conservation")
        assert cons_check.state == ValidationState.VALID
