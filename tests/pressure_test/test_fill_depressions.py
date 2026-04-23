"""FillDepressions operator pressure test.

Lane: operator

Stress points:
1. Single-celled pits filled correctly
2. Multi-celled depressions filled to spill elevation
3. Nodata cells preserved and not filled
4. Boundary cells treated as outlets (never raised)
5. Elevation only raised, never lowered
6. Flat gradient resolution enables D8 flow
7. Already-drained DEM passes through unchanged
8. Operator protocol compliance (spec, validate_inputs, declared_checks)
9. OperatorResult contains valid artifact with fresh metadata
10. Lineage records algorithm params
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
from quarry_operators.fill_depressions import (
    FillDepressionsOperator,
    FillDepressionsParams,
    _count_interior_pits,
    _priority_flood,
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


def _make_artifact(path, nodata=None):
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
    return FillDepressionsOperator()


@pytest.fixture
def single_pit_dem(tmp_path):
    """5x5 DEM with a single pit in the center, boundary low enough to drain ring."""
    # Boundary at 5, ring at 7, pit at 3.
    # Ring drains freely (7 > 5 boundary). Only center pit (3) fills to ring level (7).
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
    path = tmp_path / "single_pit.tif"
    _write_dem(path, dem)
    return path, dem


@pytest.fixture
def multi_cell_depression(tmp_path):
    """7x7 DEM with a multi-cell depression, one low boundary outlet."""
    # Bowl interior at 3-5, rim at 8, boundary at 10 except one cell at 6.
    # Spill elevation = 8 (the rim), because boundary outlet at 6 lets rim drain.
    dem = np.array(
        [
            [10, 10, 10, 10, 10, 10, 10],
            [10, 8, 8, 8, 8, 8, 10],
            [10, 8, 5, 5, 5, 8, 10],
            [6, 8, 5, 3, 5, 8, 10],
            [10, 8, 5, 5, 5, 8, 10],
            [10, 8, 8, 8, 8, 8, 10],
            [10, 10, 10, 10, 10, 10, 10],
        ],
        dtype=np.float64,
    )
    path = tmp_path / "multi_cell.tif"
    _write_dem(path, dem)
    return path, dem


@pytest.fixture
def sloped_dem(tmp_path):
    """10x10 DEM that already drains — no depressions."""
    rows = np.arange(10, dtype=np.float64).reshape(10, 1)
    dem = np.broadcast_to(rows, (10, 10)).copy()
    path = tmp_path / "sloped.tif"
    _write_dem(path, dem)
    return path, dem


@pytest.fixture
def nodata_dem(tmp_path):
    """DEM with nodata hole in the middle."""
    dem = np.full((7, 7), 10.0, dtype=np.float64)
    dem[3, 3] = -9999.0  # nodata cell
    dem[2, 3] = 5.0  # pit next to nodata
    path = tmp_path / "nodata.tif"
    _write_dem(path, dem, nodata=-9999.0)
    return path, dem


@pytest.fixture
def channel_dem(tmp_path):
    """DEM with a channel that drains to the edge — depression beside it."""
    dem = np.full((9, 9), 10.0, dtype=np.float64)
    # Channel from center to south edge
    for r in range(4, 9):
        dem[r, 4] = 5.0
    # Depression beside channel
    dem[4, 2] = 3.0
    dem[4, 3] = 4.0
    dem[5, 2] = 4.0
    dem[5, 3] = 3.0
    path = tmp_path / "channel.tif"
    _write_dem(path, dem)
    return path, dem


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


class TestProtocolCompliance:
    def test_implements_operator_protocol(self, op):
        assert isinstance(op, Operator)

    def test_name(self, op):
        assert op.name == "fill_depressions"

    def test_spec_shape(self, op):
        spec = op.spec
        assert spec.input_types == (ArtifactType.RASTER,)
        assert spec.output_type == ArtifactType.RASTER
        assert spec.min_inputs == 1
        assert spec.max_inputs == 1
        assert spec.resource_scale == ResourceScale.MEDIUM

    def test_declared_checks(self, op):
        checks = op.declared_checks()
        assert "no_interior_pits" in checks
        assert "elevation_only_raised" in checks
        assert "backing_accessible" in checks


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_rejects_no_inputs(self, op):
        errors = op.validate_inputs([], FillDepressionsParams(output_path="/tmp/x.tif"))
        assert any("required" in e.lower() for e in errors)

    def test_rejects_vector_input(self, op, tmp_path):
        art = Artifact(
            type=ArtifactType.VECTOR,
            name="v",
            backing=BackingStore(
                kind=BackingStoreKind.LOCAL_FILE, uri="/fake", size_bytes=0, content_hash=""
            ),
        )
        errors = op.validate_inputs([art], FillDepressionsParams(output_path="/tmp/x.tif"))
        assert any("raster" in e.lower() for e in errors)

    def test_rejects_missing_output_path(self, op, single_pit_dem):
        path, _ = single_pit_dem
        art = _make_artifact(path)
        errors = op.validate_inputs([art], FillDepressionsParams(output_path=""))
        assert any("output_path" in e for e in errors)

    def test_rejects_bad_epsilon(self, op, single_pit_dem):
        path, _ = single_pit_dem
        art = _make_artifact(path)
        errors = op.validate_inputs(
            [art], FillDepressionsParams(output_path="/tmp/x.tif", epsilon=-1.0)
        )
        assert any("epsilon" in e for e in errors)

    def test_accepts_valid_input(self, op, single_pit_dem):
        path, _ = single_pit_dem
        art = _make_artifact(path)
        errors = op.validate_inputs([art], FillDepressionsParams(output_path="/tmp/x.tif"))
        assert errors == []


# ---------------------------------------------------------------------------
# Algorithm correctness
# ---------------------------------------------------------------------------


class TestPriorityFlood:
    def test_single_pit_filled_to_spill(self, single_pit_dem):
        """Single pit at center should be raised to rim elevation (7)."""
        _, dem = single_pit_dem
        valid = np.ones(dem.shape, dtype=bool)
        filled = _priority_flood(dem, valid)
        # Center pit (3) should be raised to 7 (the rim)
        assert filled[2, 2] == 7.0

    def test_multi_cell_depression_filled(self, multi_cell_depression):
        """All cells in the bowl should be raised to spill elevation (8)."""
        _, dem = multi_cell_depression
        valid = np.ones(dem.shape, dtype=bool)
        filled = _priority_flood(dem, valid)
        # Inner cells (5, 3) should all be raised to 8 (the rim of the bowl)
        assert filled[3, 3] == 8.0
        assert filled[2, 2] == 8.0
        assert filled[4, 4] == 8.0

    def test_sloped_dem_unchanged(self, sloped_dem):
        """A DEM with no depressions should pass through unchanged."""
        _, dem = sloped_dem
        valid = np.ones(dem.shape, dtype=bool)
        filled = _priority_flood(dem, valid)
        np.testing.assert_array_equal(filled, dem)

    def test_nodata_not_filled(self, nodata_dem):
        """Nodata cells should not be modified."""
        _, dem = nodata_dem
        valid = dem != -9999.0
        filled = _priority_flood(dem, valid)
        assert filled[3, 3] == -9999.0  # nodata preserved

    def test_boundary_never_raised(self, single_pit_dem):
        """Boundary cells are outlets — they must never be raised."""
        _, dem = single_pit_dem
        valid = np.ones(dem.shape, dtype=bool)
        original_boundary = dem[0, :].copy()
        filled = _priority_flood(dem, valid)
        np.testing.assert_array_equal(filled[0, :], original_boundary)

    def test_elevation_only_increases(self, multi_cell_depression):
        """No cell should be lowered."""
        _, dem = multi_cell_depression
        valid = np.ones(dem.shape, dtype=bool)
        filled = _priority_flood(dem, valid)
        assert np.all(filled[valid] >= dem[valid])

    def test_no_pits_remain(self, multi_cell_depression):
        """After filling, no interior pits should exist."""
        _, dem = multi_cell_depression
        valid = np.ones(dem.shape, dtype=bool)
        filled = _priority_flood(dem, valid)
        assert _count_interior_pits(filled, valid) == 0

    def test_channel_depression_fills_to_channel_level(self, channel_dem):
        """Depression beside a channel should fill to channel elevation, not rim."""
        _, dem = channel_dem
        valid = np.ones(dem.shape, dtype=bool)
        filled = _priority_flood(dem, valid)
        # The depression cells (3, 4 elevation) should fill to 5 (channel level)
        # because the channel provides a lower spill path than the rim (10)
        assert filled[4, 2] == 5.0
        assert filled[5, 3] == 5.0


# ---------------------------------------------------------------------------
# Full operator execution
# ---------------------------------------------------------------------------


class TestExecution:
    def test_execute_single_pit(self, op, single_pit_dem, tmp_path):
        path, dem = single_pit_dem
        art = _make_artifact(path)
        output = tmp_path / "filled.tif"
        params = FillDepressionsParams(output_path=str(output))

        result = op.execute([art], params)

        assert result.artifact.type == ArtifactType.RASTER
        assert result.artifact.backing.uri == str(output)
        assert output.exists()

        # Verify filled data
        with rasterio.open(output) as src:
            data = src.read(1)
            assert data[2, 2] == 7.0  # pit filled

    def test_execute_preserves_crs(self, op, single_pit_dem, tmp_path):
        path, _ = single_pit_dem
        art = _make_artifact(path)
        output = tmp_path / "filled.tif"
        params = FillDepressionsParams(output_path=str(output))

        result = op.execute([art], params)
        assert "32610" in result.artifact.spatial.crs

    def test_execute_lineage(self, op, single_pit_dem, tmp_path):
        path, _ = single_pit_dem
        art = _make_artifact(path)
        output = tmp_path / "filled.tif"
        params = FillDepressionsParams(output_path=str(output))

        result = op.execute([art], params)
        assert result.artifact.lineage.operation == "fill_depressions"
        assert "epsilon" in result.artifact.lineage.params
        assert "apply_gradient" in result.artifact.lineage.params

    def test_execute_checks_all_pass(self, op, single_pit_dem, tmp_path):
        path, _ = single_pit_dem
        art = _make_artifact(path)
        output = tmp_path / "filled.tif"
        params = FillDepressionsParams(output_path=str(output))

        result = op.execute([art], params)
        for check in result.checks:
            assert check.state == ValidationState.VALID, f"{check.check_name}: {check.message}"

    def test_execute_nodata_preserved(self, op, nodata_dem, tmp_path):
        path, _ = nodata_dem
        art = _make_artifact(path)
        output = tmp_path / "filled.tif"
        params = FillDepressionsParams(output_path=str(output), nodata=-9999.0)

        op.execute([art], params)
        with rasterio.open(output) as src:
            data = src.read(1)
            assert data[3, 3] == -9999.0

    def test_execute_metadata_fresh(self, op, single_pit_dem, tmp_path):
        """Output metadata must come from actual file, not copied."""
        path, _ = single_pit_dem
        art = _make_artifact(path)
        output = tmp_path / "filled.tif"
        params = FillDepressionsParams(output_path=str(output))

        result = op.execute([art], params)
        assert result.artifact.backing.size_bytes > 0
        assert result.artifact.backing.content_hash != ""
        assert result.artifact.metadata["algorithm"] == "priority_flood_wang_liu_2006"

    def test_execute_sloped_passthrough(self, op, sloped_dem, tmp_path):
        """Already-drained DEM should produce identical output."""
        path, dem = sloped_dem
        art = _make_artifact(path)
        output = tmp_path / "filled.tif"
        params = FillDepressionsParams(output_path=str(output), apply_gradient=False)

        op.execute([art], params)
        with rasterio.open(output) as src:
            data = src.read(1)
            np.testing.assert_array_almost_equal(data, dem)


# ---------------------------------------------------------------------------
# Flat gradient resolution
# ---------------------------------------------------------------------------


class TestFlatGradient:
    def test_gradient_creates_slope_in_flat(self, multi_cell_depression, tmp_path):
        """After filling + gradient, flat region cells should have slight differences."""
        _, dem = multi_cell_depression
        op = FillDepressionsOperator()
        path = tmp_path / "bowl.tif"
        _write_dem(path, dem)
        art = _make_artifact(path)
        output = tmp_path / "filled_grad.tif"
        params = FillDepressionsParams(output_path=str(output), apply_gradient=True)

        op.execute([art], params)
        with rasterio.open(output) as src:
            data = src.read(1)

        # The filled flat region should NOT be perfectly flat anymore
        interior = data[2:5, 2:5]
        # With gradient, there should be slight variation
        assert interior.max() - interior.min() > 0

    def test_gradient_disabled_keeps_flat(self, multi_cell_depression, tmp_path):
        """Without gradient, filled region stays flat."""
        _, dem = multi_cell_depression
        path = tmp_path / "bowl.tif"
        _write_dem(path, dem)
        art = _make_artifact(path)
        output = tmp_path / "filled_flat.tif"
        params = FillDepressionsParams(output_path=str(output), apply_gradient=False)

        op = FillDepressionsOperator()
        op.execute([art], params)
        with rasterio.open(output) as src:
            data = src.read(1)

        # Filled region should be perfectly flat at spill elevation
        assert data[2, 2] == data[3, 3] == data[4, 4]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_tiny_1x1(self, tmp_path):
        """1x1 DEM — single boundary cell, nothing to fill."""
        dem = np.array([[5.0]])
        path = tmp_path / "tiny.tif"
        _write_dem(path, dem)
        art = _make_artifact(path)
        output = tmp_path / "filled.tif"
        params = FillDepressionsParams(output_path=str(output), apply_gradient=False)

        op = FillDepressionsOperator()
        op.execute([art], params)
        with rasterio.open(output) as src:
            data = src.read(1)
            assert data[0, 0] == 5.0

    def test_all_same_elevation(self, tmp_path):
        """Flat DEM — no depressions, no changes."""
        dem = np.full((5, 5), 7.0, dtype=np.float64)
        path = tmp_path / "flat.tif"
        _write_dem(path, dem)
        art = _make_artifact(path)
        output = tmp_path / "filled.tif"
        params = FillDepressionsParams(output_path=str(output), apply_gradient=False)

        op = FillDepressionsOperator()
        op.execute([art], params)
        with rasterio.open(output) as src:
            data = src.read(1)
            np.testing.assert_array_equal(data, dem)

    def test_nested_depressions(self, tmp_path):
        """Two nested depressions at different depths."""
        dem = np.full((9, 9), 10.0, dtype=np.float64)
        # Outer depression
        dem[2:7, 2:7] = 6.0
        # Inner deeper depression
        dem[3:6, 3:6] = 3.0
        path = tmp_path / "nested.tif"
        _write_dem(path, dem)

        valid = np.ones(dem.shape, dtype=bool)
        filled = _priority_flood(dem, valid)

        # Everything should fill to rim (10)
        assert filled[4, 4] == 10.0
        assert filled[2, 2] == 10.0
        assert _count_interior_pits(filled, valid) == 0

    def test_large_random_dem(self, tmp_path):
        """Random 100x100 DEM — correctness property: no pits after fill."""
        rng = np.random.default_rng(42)
        dem = rng.uniform(0, 100, size=(100, 100))
        path = tmp_path / "random.tif"
        _write_dem(path, dem.astype(np.float64))
        art = _make_artifact(path)
        output = tmp_path / "filled.tif"
        params = FillDepressionsParams(output_path=str(output), apply_gradient=False)

        op = FillDepressionsOperator()
        op.execute([art], params)

        with rasterio.open(output) as src:
            data = src.read(1)

        valid = np.ones(data.shape, dtype=bool)
        assert _count_interior_pits(data, valid) == 0
        assert np.all(data >= dem)
