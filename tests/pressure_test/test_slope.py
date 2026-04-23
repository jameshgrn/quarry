"""Slope operator pressure test.

Stress points:
1. Perfectly flat DEM produces zero slope everywhere
2. 45-degree inclined plane produces correct slope (45° / 100% / π/4 rad)
3. Nodata cells produce nodata in output
4. Resolution correctly applied (not assuming 1x1 cells)
5. All three unit conversions (degrees, percent, radians) correct
6. Steep slope (>45°) produces correct values
7. Operator protocol compliance (spec, validate_inputs, declared_checks)
8. OperatorResult contains valid artifact with fresh metadata
9. Lineage records algorithm params
10. Central difference gradient accuracy on synthetic surfaces
"""

import math

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
from quarry_operators.slope import SlopeOperator, SlopeParams
from rasterio.crs import CRS
from rasterio.transform import from_origin

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_dem(path, data, transform=None, nodata=None, crs=None):
    """Write a single-band DEM GeoTIFF."""
    nrows, ncols = data.shape
    if transform is None:
        # Default: 1m cells, origin at (0,0)
        transform = from_origin(0, nrows, 1, 1)
    if crs is None:
        crs = CRS.from_epsg(32610)  # UTM 10N, meters

    meta = {
        "driver": "GTiff",
        "height": nrows,
        "width": ncols,
        "count": 1,
        "dtype": str(data.dtype),
        "crs": crs,
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
            ),
        )


@pytest.fixture
def op():
    return SlopeOperator()


@pytest.fixture
def flat_dem(tmp_path):
    """Perfectly flat 10x10 DEM at elevation 100."""
    dem = np.full((10, 10), 100.0, dtype=np.float64)
    path = tmp_path / "flat.tif"
    _write_dem(path, dem)
    return path, dem


@pytest.fixture
def inclined_plane_45(tmp_path):
    """DEM inclined at exactly 45 degrees (1:1 rise:run)."""
    # 10x10 grid, 1m cells. Elevation rises 1m per cell in X direction.
    # Slope should be 45° = 100% = π/4 radians everywhere.
    dem = np.zeros((10, 10), dtype=np.float64)
    for c in range(10):
        dem[:, c] = c  # Elevation = column index
    path = tmp_path / "inclined_45.tif"
    _write_dem(path, dem)
    return path, dem


@pytest.fixture
def steep_slope(tmp_path):
    """DEM with 2:1 rise (about 63.4 degrees)."""
    dem = np.zeros((10, 10), dtype=np.float64)
    for c in range(10):
        dem[:, c] = 2.0 * c  # 2m rise per 1m run
    path = tmp_path / "steep.tif"
    _write_dem(path, dem)
    return path, dem


@pytest.fixture
def nodata_dem(tmp_path):
    """DEM with nodata hole."""
    dem = np.full((10, 10), 100.0, dtype=np.float64)
    dem[4:6, 4:6] = -9999.0
    path = tmp_path / "nodata.tif"
    _write_dem(path, dem, nodata=-9999.0)
    return path, dem


@pytest.fixture
def parabolic_surface(tmp_path):
    """Parabolic surface: z = x^2 + y^2. Known gradient at any point."""
    x = np.linspace(-5, 5, 101)
    y = np.linspace(-5, 5, 101)
    X, Y = np.meshgrid(x, y)
    dem = X**2 + Y**2
    # At (3, 0): dz/dx = 6, dz/dy = 0, slope = atan(6) ≈ 80.54°
    # At (0, 0): slope = 0
    path = tmp_path / "parabola.tif"
    _write_dem(path, dem.astype(np.float64))
    return path, dem, (x, y, X, Y)


@pytest.fixture
def large_cell_dem(tmp_path):
    """DEM with 10m cells (not 1m). Tests resolution handling."""
    dem = np.zeros((10, 10), dtype=np.float64)
    for c in range(10):
        dem[:, c] = c  # 1m rise per column
    # Transform: 10m x 10m cells
    transform = from_origin(0, 100, 10, 10)
    path = tmp_path / "large_cell.tif"
    _write_dem(path, dem, transform=transform)
    return path, dem


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


class TestProtocolCompliance:
    def test_implements_operator_protocol(self, op):
        assert isinstance(op, Operator)

    def test_name(self, op):
        assert op.name == "slope"

    def test_spec_shape(self, op):
        spec = op.spec
        assert spec.input_types == (ArtifactType.RASTER,)
        assert spec.output_type == ArtifactType.RASTER
        assert spec.min_inputs == 1
        assert spec.max_inputs == 1
        assert spec.resource_scale == ResourceScale.MEDIUM

    def test_declared_checks(self, op):
        checks = op.declared_checks()
        assert "valid_range" in checks
        assert "nodata_preserved" in checks
        assert "resolution_consistent" in checks


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_rejects_no_inputs(self, op):
        errors = op.validate_inputs([], SlopeParams(output_path="/tmp/x.tif"))
        assert any("required" in e.lower() for e in errors)

    def test_rejects_vector_input(self, op, tmp_path):
        art = Artifact(
            type=ArtifactType.VECTOR,
            name="v",
            backing=BackingStore(
                kind=BackingStoreKind.LOCAL_FILE, uri="/fake", size_bytes=0, content_hash=""
            ),
        )
        errors = op.validate_inputs([art], SlopeParams(output_path="/tmp/x.tif"))
        assert any("raster" in e.lower() for e in errors)

    def test_rejects_missing_output_path(self, op, flat_dem):
        path, _ = flat_dem
        art = _make_artifact(path)
        errors = op.validate_inputs([art], SlopeParams(output_path=""))
        assert any("output_path" in e for e in errors)

    def test_rejects_invalid_units(self, op, flat_dem):
        path, _ = flat_dem
        art = _make_artifact(path)
        errors = op.validate_inputs([art], SlopeParams(output_path="/tmp/x.tif", units="feet"))
        assert any("units" in e.lower() for e in errors)

    def test_accepts_valid_input(self, op, flat_dem):
        path, _ = flat_dem
        art = _make_artifact(path)
        errors = op.validate_inputs([art], SlopeParams(output_path="/tmp/x.tif"))
        assert errors == []


# ---------------------------------------------------------------------------
# Algorithm correctness
# ---------------------------------------------------------------------------


class TestFlatSurface:
    def test_flat_produces_zero_slope(self, op, flat_dem, tmp_path):
        """Perfectly flat DEM should produce zero slope everywhere."""
        path, _ = flat_dem
        art = _make_artifact(path)
        output = tmp_path / "slope.tif"
        params = SlopeParams(output_path=str(output), units="degrees")

        result = op.execute([art], params)

        with rasterio.open(output) as src:
            slope = src.read(1)
            valid = slope != params.output_nodata
            assert np.all(slope[valid] == 0.0)

    def test_flat_all_units_zero(self, op, flat_dem, tmp_path):
        """Flat surface: all unit conversions should give zero/valid."""
        path, _ = flat_dem
        art = _make_artifact(path)

        for units in ["degrees", "percent", "radians"]:
            output = tmp_path / f"slope_{units}.tif"
            params = SlopeParams(output_path=str(output), units=units)
            op.execute([art], params)

            with rasterio.open(output) as src:
                slope = src.read(1)
                valid = slope != params.output_nodata
                assert np.all(slope[valid] == 0.0), f"Failed for {units}"


class TestInclinedPlane:
    def test_45_degree_in_degrees(self, op, inclined_plane_45, tmp_path):
        """45° incline should produce ~45° slope."""
        path, _ = inclined_plane_45
        art = _make_artifact(path)
        output = tmp_path / "slope.tif"
        params = SlopeParams(output_path=str(output), units="degrees")

        op.execute([art], params)

        with rasterio.open(output) as src:
            slope = src.read(1)
            valid = slope != params.output_nodata
            # Interior cells (not edges) should be ~45°
            interior = slope[1:-1, 1:-1]
            # Central difference on inclined plane: exactly 45°
            np.testing.assert_allclose(interior, 45.0, rtol=1e-10)

    def test_45_degree_in_percent(self, op, inclined_plane_45, tmp_path):
        """45° incline = 100% slope."""
        path, _ = inclined_plane_45
        art = _make_artifact(path)
        output = tmp_path / "slope.tif"
        params = SlopeParams(output_path=str(output), units="percent")

        op.execute([art], params)

        with rasterio.open(output) as src:
            slope = src.read(1)
            interior = slope[1:-1, 1:-1]
            np.testing.assert_allclose(interior, 100.0, rtol=1e-10)

    def test_45_degree_in_radians(self, op, inclined_plane_45, tmp_path):
        """45° incline = π/4 radians."""
        path, _ = inclined_plane_45
        art = _make_artifact(path)
        output = tmp_path / "slope.tif"
        params = SlopeParams(output_path=str(output), units="radians")

        op.execute([art], params)

        with rasterio.open(output) as src:
            slope = src.read(1)
            interior = slope[1:-1, 1:-1]
            np.testing.assert_allclose(interior, math.pi / 4, rtol=1e-10)


class TestSteepSlope:
    def test_63_degree_slope(self, op, steep_slope, tmp_path):
        """2:1 rise should give arctan(2) ≈ 63.43°."""
        path, _ = steep_slope
        art = _make_artifact(path)
        output = tmp_path / "slope.tif"
        params = SlopeParams(output_path=str(output), units="degrees")

        op.execute([art], params)

        with rasterio.open(output) as src:
            slope = src.read(1)
            interior = slope[1:-1, 1:-1]
            expected = math.degrees(math.atan(2.0))
            np.testing.assert_allclose(interior, expected, rtol=1e-10)


class TestParabolicSurface:
    def test_parabola_center_zero(self, op, parabolic_surface, tmp_path):
        """At center of parabola, slope should be zero."""
        path, _, _ = parabolic_surface
        art = _make_artifact(path)
        output = tmp_path / "slope.tif"
        params = SlopeParams(output_path=str(output), units="degrees")

        op.execute([art], params)

        with rasterio.open(output) as src:
            slope = src.read(1)
            # Center is at (50, 50) in 101x101 grid
            center_slope = slope[50, 50]
            assert abs(center_slope) < 0.01  # Near zero

    def test_parabola_gradient_increases_away_from_center(self, op, parabolic_surface, tmp_path):
        """Slope should increase with distance from center."""
        path, _, (x, y, X, Y) = parabolic_surface
        art = _make_artifact(path)
        output = tmp_path / "slope.tif"
        params = SlopeParams(output_path=str(output), units="degrees")

        op.execute([art], params)

        with rasterio.open(output) as src:
            slope = src.read(1)
            # Sample along positive X axis
            slopes_along_x = slope[50, 50:60]
            # Should be monotonically increasing
            for i in range(len(slopes_along_x) - 1):
                assert slopes_along_x[i + 1] > slopes_along_x[i]


class TestResolutionHandling:
    def test_large_cells_produce_same_slope(self, op, large_cell_dem, tmp_path):
        """10m cells with same rise/run ratio should give same slope."""
        path, _ = large_cell_dem
        art = _make_artifact(path)
        output = tmp_path / "slope.tif"
        params = SlopeParams(output_path=str(output), units="degrees")

        op.execute([art], params)

        with rasterio.open(output) as src:
            slope = src.read(1)
            interior = slope[1:-1, 1:-1]
            # 1m rise per 10m cell = 0.1 gradient = ~5.71°
            expected = math.degrees(math.atan(0.1))
            np.testing.assert_allclose(interior, expected, rtol=1e-10)


# ---------------------------------------------------------------------------
# Nodata handling
# ---------------------------------------------------------------------------


class TestNodata:
    def test_nodata_preserved(self, op, nodata_dem, tmp_path):
        """Input nodata cells should produce output nodata."""
        path, _ = nodata_dem
        art = _make_artifact(path)
        output = tmp_path / "slope.tif"
        params = SlopeParams(output_path=str(output), units="degrees", nodata=-9999.0)

        op.execute([art], params)

        with rasterio.open(output) as src:
            slope = src.read(1)
            # The 2x2 nodata region
            assert slope[4, 4] == params.output_nodata
            assert slope[4, 5] == params.output_nodata
            assert slope[5, 4] == params.output_nodata
            assert slope[5, 5] == params.output_nodata

    def test_nodata_check_passes(self, op, nodata_dem, tmp_path):
        """Nodata preservation check should pass."""
        path, _ = nodata_dem
        art = _make_artifact(path)
        output = tmp_path / "slope.tif"
        params = SlopeParams(output_path=str(output), units="degrees", nodata=-9999.0)

        result = op.execute([art], params)

        nodata_check = next(c for c in result.checks if c.check_name == "nodata_preserved")
        assert nodata_check.state == ValidationState.VALID


# ---------------------------------------------------------------------------
# Full operator execution
# ---------------------------------------------------------------------------


class TestExecution:
    def test_execute_produces_raster(self, op, inclined_plane_45, tmp_path):
        path, _ = inclined_plane_45
        art = _make_artifact(path)
        output = tmp_path / "slope.tif"
        params = SlopeParams(output_path=str(output))

        result = op.execute([art], params)

        assert result.artifact.type == ArtifactType.RASTER
        assert result.artifact.backing.uri == str(output)
        assert output.exists()

    def test_execute_preserves_crs(self, op, inclined_plane_45, tmp_path):
        path, _ = inclined_plane_45
        art = _make_artifact(path)
        output = tmp_path / "slope.tif"
        params = SlopeParams(output_path=str(output))

        result = op.execute([art], params)
        assert "32610" in result.artifact.spatial.crs

    def test_execute_lineage(self, op, inclined_plane_45, tmp_path):
        path, _ = inclined_plane_45
        art = _make_artifact(path)
        output = tmp_path / "slope.tif"
        params = SlopeParams(output_path=str(output), units="degrees")

        result = op.execute([art], params)
        assert result.artifact.lineage.operation == "slope"
        assert result.artifact.lineage.params["units"] == "degrees"

    def test_execute_metadata(self, op, inclined_plane_45, tmp_path):
        path, _ = inclined_plane_45
        art = _make_artifact(path)
        output = tmp_path / "slope.tif"
        params = SlopeParams(output_path=str(output), units="degrees")

        result = op.execute([art], params)
        assert result.artifact.metadata["algorithm"] == "central_difference_gradient"
        assert result.artifact.metadata["units"] == "degrees"

    def test_execute_checks_pass(self, op, inclined_plane_45, tmp_path):
        path, _ = inclined_plane_45
        art = _make_artifact(path)
        output = tmp_path / "slope.tif"
        params = SlopeParams(output_path=str(output))

        result = op.execute([art], params)
        for check in result.checks:
            assert check.state == ValidationState.VALID, f"{check.check_name}: {check.message}"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_tiny_3x3(self, op, tmp_path):
        """Minimum viable grid for central difference."""
        # Use a plane instead of pyramid — central diff on pyramid gives 0 at center
        dem = np.array(
            [
                [0.0, 1.0, 2.0],
                [0.0, 1.0, 2.0],
                [0.0, 1.0, 2.0],
            ],
            dtype=np.float64,
        )
        path = tmp_path / "tiny.tif"
        _write_dem(path, dem)
        art = _make_artifact(path)
        output = tmp_path / "slope.tif"
        params = SlopeParams(output_path=str(output), units="degrees")

        op.execute([art], params)
        with rasterio.open(output) as src:
            slope = src.read(1)
            # Interior cells should have slope ~45° (1m rise per 1m run)
            assert slope[1, 1] > 40  # approximately 45°

    def test_single_row(self, op, tmp_path):
        """1xN grid — gradient only in X direction, Y gradient is zero."""
        dem = np.arange(10, dtype=np.float64).reshape(1, 10)
        path = tmp_path / "single_row.tif"
        _write_dem(path, dem)
        art = _make_artifact(path)
        output = tmp_path / "slope.tif"
        params = SlopeParams(output_path=str(output), units="degrees")

        result = op.execute([art], params)
        assert result.artifact is not None

        # Verify slope is calculated from X gradient only
        with rasterio.open(output) as src:
            slope = src.read(1)
            # 1m rise per 1m cell = 45°
            # Central difference on interior: exactly 45°
            np.testing.assert_allclose(slope[0, 1:-1], 45.0, rtol=1e-10)

    def test_all_nodata(self, op, tmp_path):
        """Completely nodata grid."""
        dem = np.full((5, 5), -9999.0, dtype=np.float64)
        path = tmp_path / "all_nodata.tif"
        _write_dem(path, dem, nodata=-9999.0)
        art = _make_artifact(path)
        output = tmp_path / "slope.tif"
        params = SlopeParams(output_path=str(output), nodata=-9999.0)

        result = op.execute([art], params)

        with rasterio.open(output) as src:
            slope = src.read(1)
            # Should be all nodata
            assert np.all(slope == params.output_nodata)

    def test_very_small_epsilon(self, op, tmp_path):
        """Near-flat surface with tiny variations."""
        dem = np.full((10, 10), 100.0, dtype=np.float64)
        dem[:, 5:] = 100.0 + 1e-6  # Tiny step
        path = tmp_path / "epsilon.tif"
        _write_dem(path, dem)
        art = _make_artifact(path)
        output = tmp_path / "slope.tif"
        params = SlopeParams(output_path=str(output), units="degrees")

        op.execute([art], params)
        with rasterio.open(output) as src:
            slope = src.read(1)
            valid = slope != params.output_nodata
            # Should detect the tiny slope
            assert np.any(slope[valid] > 0)
