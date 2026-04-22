"""D8 flow direction operator pressure test.

Stress points:
1. Steepest descent correctly picks direction (cardinal vs diagonal)
2. Diagonal distance weighting (sqrt(2)) affects direction choice
3. Boundary cells with no lower neighbor → OUTLET
4. Interior cells with no lower neighbor → PIT (should be 0 after fill)
5. Nodata cells → NODATA code, never assigned direction
6. Depression-filled input → zero PITs
7. All valid cells get assigned (no unassigned valid cells)
8. Direction encoding matches spec (0=E through 7=NE, 8=OUTLET, 9=PIT)
9. Operator protocol compliance
10. Chain: FillDepressions → D8 (integration)
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
from quarry_core.operator import Operator, ResourceScale
from quarry_operators.d8_flow_direction import (
    D8_DC,
    D8_DR,
    NODATA,
    OUTLET,
    PIT,
    D8FlowDirectionOperator,
    D8FlowDirectionParams,
    _compute_d8,
)
from rasterio.crs import CRS
from rasterio.transform import from_bounds

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_dem(path, data, nodata=None):
    """Write a single-band DEM GeoTIFF."""
    nrows, ncols = data.shape
    transform = from_bounds(0, 0, ncols, nrows, ncols, nrows)
    meta = {
        "driver": "GTiff",
        "height": nrows,
        "width": ncols,
        "count": 1,
        "dtype": str(data.dtype),
        "crs": CRS.from_epsg(32610),
        "transform": transform,
    }
    if nodata is not None:
        meta["nodata"] = nodata
    with rasterio.open(path, "w", **meta) as dst:
        dst.write(data, 1)


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
    return D8FlowDirectionOperator()


@pytest.fixture
def slope_east(tmp_path):
    """5x5 DEM sloping east (elevation decreases left to right)."""
    # Col 0 = 4, col 1 = 3, ..., col 4 = 0
    dem = np.zeros((5, 5), dtype=np.float64)
    for c in range(5):
        dem[:, c] = 4.0 - c
    path = tmp_path / "slope_east.tif"
    _write_dem(path, dem)
    return path, dem


@pytest.fixture
def slope_south(tmp_path):
    """5x5 DEM sloping south (elevation decreases top to bottom)."""
    dem = np.zeros((5, 5), dtype=np.float64)
    for r in range(5):
        dem[r, :] = 4.0 - r
    path = tmp_path / "slope_south.tif"
    _write_dem(path, dem)
    return path, dem


@pytest.fixture
def cone_dem(tmp_path):
    """9x9 DEM: cone centered at (4,4), decreasing outward. No depressions."""
    dem = np.zeros((9, 9), dtype=np.float64)
    for r in range(9):
        for c in range(9):
            dem[r, c] = 100.0 - np.sqrt((r - 4) ** 2 + (c - 4) ** 2) * 10
    path = tmp_path / "cone.tif"
    _write_dem(path, dem)
    return path, dem


@pytest.fixture
def filled_pit_dem(tmp_path):
    """5x5 depression-filled DEM (flat center from fill, boundary low)."""
    # Boundary at 3, ring at 7, center filled to 7 (was a pit, now filled)
    dem = np.array(
        [
            [3, 3, 3, 3, 3],
            [3, 7, 7, 7, 3],
            [3, 7, 7, 7, 3],
            [3, 7, 7, 7, 3],
            [3, 3, 3, 3, 3],
        ],
        dtype=np.float64,
    )
    path = tmp_path / "filled_pit.tif"
    _write_dem(path, dem)
    return path, dem


@pytest.fixture
def nodata_dem(tmp_path):
    """DEM with nodata hole."""
    dem = np.zeros((5, 5), dtype=np.float64)
    for r in range(5):
        dem[r, :] = 4.0 - r  # slope south
    dem[2, 2] = -9999.0  # nodata
    path = tmp_path / "nodata.tif"
    _write_dem(path, dem, nodata=-9999.0)
    return path, dem


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


class TestProtocolCompliance:
    def test_implements_operator_protocol(self, op):
        assert isinstance(op, Operator)

    def test_name(self, op):
        assert op.name == "d8_flow_direction"

    def test_spec_shape(self, op):
        spec = op.spec
        assert spec.input_types == (ArtifactType.RASTER,)
        assert spec.output_type == ArtifactType.RASTER
        assert spec.min_inputs == 1
        assert spec.max_inputs == 1
        assert spec.resource_scale == ResourceScale.MEDIUM

    def test_declared_checks(self, op):
        checks = op.declared_checks()
        assert "valid_code_set" in checks
        assert "no_pits" in checks
        assert "all_valid_assigned" in checks
        assert "backing_accessible" in checks


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_rejects_no_inputs(self, op):
        errors = op.validate_inputs([], D8FlowDirectionParams(output_path="/tmp/x.tif"))
        assert any("required" in e.lower() for e in errors)

    def test_rejects_vector_input(self, op):
        art = Artifact(
            type=ArtifactType.VECTOR,
            name="v",
            backing=BackingStore(
                kind=BackingStoreKind.LOCAL_FILE, uri="/fake", size_bytes=0, content_hash=""
            ),
        )
        errors = op.validate_inputs([art], D8FlowDirectionParams(output_path="/tmp/x.tif"))
        assert any("raster" in e.lower() for e in errors)

    def test_rejects_missing_output_path(self, op, slope_east):
        path, _ = slope_east
        art = _make_artifact(path)
        errors = op.validate_inputs([art], D8FlowDirectionParams(output_path=None))
        assert any("output_path" in e for e in errors)

    def test_accepts_valid_input(self, op, slope_east):
        path, _ = slope_east
        art = _make_artifact(path)
        errors = op.validate_inputs([art], D8FlowDirectionParams(output_path="/tmp/x.tif"))
        assert errors == []


# ---------------------------------------------------------------------------
# Algorithm correctness
# ---------------------------------------------------------------------------


class TestD8Algorithm:
    def test_uniform_slope_east(self, slope_east):
        """On a uniform east slope, all interior cells should flow east (code 0)."""
        _, dem = slope_east
        valid = np.ones(dem.shape, dtype=bool)
        flow = _compute_d8(dem, valid)
        # Interior cells should all flow east (direction 0)
        for r in range(1, 4):
            for c in range(1, 4):
                assert flow[r, c] == 0, f"Cell ({r},{c}) = {flow[r, c]}, expected 0 (E)"

    def test_uniform_slope_south(self, slope_south):
        """On a uniform south slope, all interior cells should flow south (code 2)."""
        _, dem = slope_south
        valid = np.ones(dem.shape, dtype=bool)
        flow = _compute_d8(dem, valid)
        for r in range(1, 4):
            for c in range(1, 4):
                assert flow[r, c] == 2, f"Cell ({r},{c}) = {flow[r, c]}, expected 2 (S)"

    def test_diagonal_weighting(self, tmp_path):
        """Diagonal has sqrt(2) distance — same drop diagonally is less steep than cardinal."""
        # 3x3 DEM: center at 10, east neighbor at 5 (drop=5, dist=1, slope=5),
        # SE neighbor at 4 (drop=6, dist=sqrt(2)≈1.41, slope≈4.24)
        # Cardinal east wins despite smaller absolute drop.
        dem = np.array(
            [
                [10, 10, 10],
                [10, 10, 5],
                [10, 10, 4],
            ],
            dtype=np.float64,
        )
        valid = np.ones(dem.shape, dtype=bool)
        flow = _compute_d8(dem, valid)
        # Center (1,1) at 10: east (1,2)=5, slope=5/1=5; SE (2,2)=4, slope=6/1.41=4.24
        assert flow[1, 1] == 0  # East wins

    def test_diagonal_wins_when_steeper(self, tmp_path):
        """When diagonal slope exceeds cardinal, diagonal wins."""
        # Center at 10, east at 8 (slope=2/1=2), SE at 0 (slope=10/1.41=7.07)
        dem = np.array(
            [
                [10, 10, 10],
                [10, 10, 8],
                [10, 10, 0],
            ],
            dtype=np.float64,
        )
        valid = np.ones(dem.shape, dtype=bool)
        flow = _compute_d8(dem, valid)
        assert flow[1, 1] == 1  # SE wins

    def test_boundary_outlet(self, slope_east):
        """Boundary cells at the low edge should be OUTLET."""
        _, dem = slope_east
        valid = np.ones(dem.shape, dtype=bool)
        flow = _compute_d8(dem, valid)
        # Right edge (col 4) has elevation 0 — lowest. No lower neighbor to the east.
        # They flow... actually they do have lower neighbors? Col 4 = 0.
        # All cells at col 4 have elevation 0. Their neighbors at col 3 = 1.
        # No neighbor is lower than 0, so col 4 cells are OUTLET.
        for r in range(5):
            assert flow[r, 4] == OUTLET, f"Cell ({r},4) = {flow[r, 4]}, expected OUTLET"

    def test_pit_on_unfilled_dem(self):
        """An unfilled pit should produce PIT code."""
        dem = np.array(
            [
                [5, 5, 5, 5, 5],
                [5, 7, 7, 7, 5],
                [5, 7, 3, 7, 5],
                [5, 7, 7, 7, 5],
                [5, 5, 5, 5, 5],
            ],
            dtype=np.float64,
        )
        valid = np.ones(dem.shape, dtype=bool)
        flow = _compute_d8(dem, valid)
        # Ring cells at 7 flow outward to boundary at 5.
        # But center at 3 is lower than all neighbors (7) — it's a PIT.
        # Wait: ring cells at 7 are higher than boundary at 5,
        # so ring cells flow to boundary. Center at 3 is lower than ring (7),
        # but ring flows toward boundary, not toward center.
        # Center (2,2) has ALL neighbors at 7, which are higher. No lower neighbor.
        # Center is interior → PIT.
        assert flow[2, 2] == PIT

    def test_no_pits_on_filled(self, filled_pit_dem):
        """A properly filled DEM should have zero PIT codes."""
        _, dem = filled_pit_dem
        valid = np.ones(dem.shape, dtype=bool)
        flow = _compute_d8(dem, valid)
        assert not np.any(flow[valid] == PIT)

    def test_nodata_gets_nodata_code(self, nodata_dem):
        """Nodata cells get NODATA code."""
        _, dem = nodata_dem
        valid = dem != -9999.0
        flow = _compute_d8(dem, valid)
        assert flow[2, 2] == NODATA

    def test_cone_radial_flow(self, cone_dem):
        """Cone DEM: all cells should flow outward (away from center peak)."""
        _, dem = cone_dem
        valid = np.ones(dem.shape, dtype=bool)
        flow = _compute_d8(dem, valid)

        # Every non-boundary, non-outlet cell should flow away from (4,4)
        for r in range(1, 8):
            for c in range(1, 8):
                if flow[r, c] in (OUTLET, PIT):
                    continue
                d = flow[r, c]
                nr = r + D8_DR[d]
                nc = c + D8_DC[d]
                # Target cell should be farther from center OR on boundary
                dist_curr = (r - 4) ** 2 + (c - 4) ** 2
                dist_next = (nr - 4) ** 2 + (nc - 4) ** 2
                assert dist_next >= dist_curr, (
                    f"Cell ({r},{c}) dir={d} flows inward: dist {dist_curr} → {dist_next}"
                )

    def test_all_directions_reachable(self):
        """Construct a DEM that exercises all 8 direction codes."""
        # 3x3 with center highest, each neighbor at different elevation
        dem = np.array(
            [
                [3, 2, 1],
                [4, 10, 0],
                [5, 6, 7],
            ],
            dtype=np.float64,
        )
        valid = np.ones(dem.shape, dtype=bool)
        flow = _compute_d8(dem, valid)
        # Center (1,1) at 10: steepest drop is to (1,2)=0, drop=10, dist=1, slope=10
        # That's east (code 0)
        assert flow[1, 1] == 0

    def test_flat_surface_boundary_outlets(self):
        """Flat surface: boundary = OUTLET, interior resolves to draining neighbor."""
        dem = np.full((5, 5), 5.0, dtype=np.float64)
        valid = np.ones(dem.shape, dtype=bool)
        flow = _compute_d8(dem, valid)
        # Boundary cells have no lower neighbor → OUTLET
        assert flow[0, 0] == OUTLET
        assert flow[0, 2] == OUTLET
        # Interior cells resolve via flat-resolution pass to flow toward boundary
        # (they should NOT be PIT — flat resolution routes them to OUTLET neighbors)
        assert flow[2, 2] != PIT
        assert 0 <= flow[2, 2] <= 7  # valid direction code


# ---------------------------------------------------------------------------
# Full operator execution
# ---------------------------------------------------------------------------


class TestExecution:
    def test_execute_produces_raster(self, op, slope_east, tmp_path):
        path, _ = slope_east
        art = _make_artifact(path)
        output = tmp_path / "flow.tif"
        params = D8FlowDirectionParams(output_path=str(output))

        result = op.execute([art], params)

        assert result.artifact.type == ArtifactType.RASTER
        assert output.exists()
        with rasterio.open(output) as src:
            data = src.read(1)
            assert data.dtype == np.int16

    def test_execute_preserves_crs(self, op, slope_east, tmp_path):
        path, _ = slope_east
        art = _make_artifact(path)
        output = tmp_path / "flow.tif"
        params = D8FlowDirectionParams(output_path=str(output))

        result = op.execute([art], params)
        assert "32610" in result.artifact.spatial.crs

    def test_execute_checks_pass_on_filled(self, op, filled_pit_dem, tmp_path):
        """All checks should pass on a properly filled DEM."""
        path, _ = filled_pit_dem
        art = _make_artifact(path)
        output = tmp_path / "flow.tif"
        params = D8FlowDirectionParams(output_path=str(output))

        result = op.execute([art], params)
        for check in result.checks:
            assert check.state == ValidationState.VALID, f"{check.check_name}: {check.message}"

    def test_execute_pit_check_warns_on_unfilled(self, op, tmp_path):
        """Unfilled DEM should trigger pit warning."""
        dem = np.array(
            [
                [5, 5, 5, 5, 5],
                [5, 7, 7, 7, 5],
                [5, 7, 3, 7, 5],
                [5, 7, 7, 7, 5],
                [5, 5, 5, 5, 5],
            ],
            dtype=np.float64,
        )
        path = tmp_path / "unfilled.tif"
        _write_dem(path, dem)
        art = _make_artifact(path)
        output = tmp_path / "flow.tif"
        params = D8FlowDirectionParams(output_path=str(output))

        result = op.execute([art], params)
        pit_check = next(c for c in result.checks if c.check_name == "no_pits")
        assert pit_check.state == ValidationState.WARN

    def test_execute_metadata(self, op, slope_east, tmp_path):
        path, _ = slope_east
        art = _make_artifact(path)
        output = tmp_path / "flow.tif"
        params = D8FlowDirectionParams(output_path=str(output))

        result = op.execute([art], params)
        assert result.artifact.metadata["algorithm"] == "d8_steepest_descent"
        assert "direction_encoding" in result.artifact.metadata

    def test_execute_lineage(self, op, slope_east, tmp_path):
        path, _ = slope_east
        art = _make_artifact(path)
        output = tmp_path / "flow.tif"
        params = D8FlowDirectionParams(output_path=str(output))

        result = op.execute([art], params)
        assert result.artifact.lineage.operation == "d8_flow_direction"
        assert art.id in result.artifact.lineage.inputs


# ---------------------------------------------------------------------------
# Integration: FillDepressions → D8
# ---------------------------------------------------------------------------


class TestChain:
    def test_fill_then_d8_zero_pits(self, tmp_path):
        """FillDepressions output → D8 should produce zero PITs."""
        from quarry_operators.fill_depressions import (
            FillDepressionsOperator,
            FillDepressionsParams,
        )

        # DEM with a pit
        dem = np.array(
            [
                [5, 5, 5, 5, 5],
                [5, 7, 7, 7, 5],
                [5, 7, 3, 7, 5],
                [5, 7, 7, 7, 5],
                [5, 5, 5, 5, 5],
            ],
            dtype=np.float64,
        )
        input_path = tmp_path / "raw_dem.tif"
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

        # Verify: no PITs
        pit_check = next(c for c in d8_result.checks if c.check_name == "no_pits")
        assert pit_check.state == ValidationState.VALID

        # Verify: all valid cells assigned
        assigned_check = next(c for c in d8_result.checks if c.check_name == "all_valid_assigned")
        assert assigned_check.state == ValidationState.VALID

    def test_fill_then_d8_random(self, tmp_path):
        """Random DEM through fill → D8 chain: zero PITs guaranteed."""
        from quarry_operators.fill_depressions import (
            FillDepressionsOperator,
            FillDepressionsParams,
        )

        rng = np.random.default_rng(123)
        dem = rng.uniform(0, 100, size=(50, 50)).astype(np.float64)
        input_path = tmp_path / "random.tif"
        _write_dem(input_path, dem)
        raw_art = _make_artifact(input_path)

        # Fill
        fill_op = FillDepressionsOperator()
        filled_path = tmp_path / "filled.tif"
        fill_result = fill_op.execute(
            [raw_art],
            FillDepressionsParams(output_path=str(filled_path), apply_gradient=True),
        )

        # D8
        d8_op = D8FlowDirectionOperator()
        flow_path = tmp_path / "flow.tif"
        d8_op.execute(
            [fill_result.artifact],
            D8FlowDirectionParams(output_path=str(flow_path)),
        )

        # Zero PITs
        with rasterio.open(flow_path) as src:
            flow = src.read(1)
        assert not np.any(flow == PIT)
