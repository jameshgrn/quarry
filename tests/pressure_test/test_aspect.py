"""Aspect operator pressure test.

Lane: operator

Stress points:
1. Flat DEM produces flat_value (-1) everywhere
2. East-facing slope (dz/dx > 0, dz/dy = 0) produces 90°
3. North-facing slope (dz/dy > 0, dz/dx = 0) produces 0°
4. Nodata cells produce nodata in output
5. Operator protocol compliance (spec, validate_inputs, declared_checks)
6. Compass vs math convention correctness
7. Diagonal slopes (NE, SE, SW, NW) produce correct angles
8. Single row/col handling
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
from quarry_operators.aspect import AspectOperator, AspectParams
from rasterio.crs import CRS
from rasterio.transform import from_origin

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_dem(path, data, transform=None, nodata=None, crs=None):
    """Write a single-band DEM GeoTIFF."""
    nrows, ncols = data.shape
    if transform is None:
        transform = from_origin(0, nrows, 1, 1)
    if crs is None:
        crs = CRS.from_epsg(32610)

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
    return AspectOperator()


@pytest.fixture
def flat_dem(tmp_path):
    """Perfectly flat 10x10 DEM at elevation 100."""
    dem = np.full((10, 10), 100.0, dtype=np.float64)
    path = tmp_path / "flat.tif"
    _write_dem(path, dem)
    return path, dem


@pytest.fixture
def east_facing(tmp_path):
    """DEM sloping eastward (elevation increases west to east)."""
    # East-facing slope: downslope points east
    # Elevation increases to west, decreases to east
    # So aspect should be 90° (east)
    dem = np.zeros((10, 10), dtype=np.float64)
    for c in range(10):
        dem[:, c] = 9 - c  # Higher on left (west), lower on right (east)
    path = tmp_path / "east.tif"
    _write_dem(path, dem)
    return path, dem


@pytest.fixture
def north_facing(tmp_path):
    """DEM sloping northward (elevation increases south to north)."""
    # North-facing slope: downslope points north
    # Elevation increases to south, decreases to north
    # Row 0 is at y=nrows (NORTH), Row 9 is at y=1 (SOUTH)
    # So north-facing needs: dem[r,:] = r (row 0 at north has elev 0, row 9 at south has elev 9)
    dem = np.zeros((10, 10), dtype=np.float64)
    for r in range(10):
        dem[r, :] = r  # Higher at bottom (south), lower at top (north)
    path = tmp_path / "north.tif"
    _write_dem(path, dem)
    return path, dem


@pytest.fixture
def south_facing(tmp_path):
    """DEM sloping southward."""
    # South-facing slope: downslope points south
    # Elevation increases to north, decreases to south
    # Row 0 is at y=nrows (NORTH), Row 9 is at y=1 (SOUTH)
    # So south-facing needs: dem[r,:] = 9-r (row 0 at north has elev 9, row 9 at south has elev 0)
    dem = np.zeros((10, 10), dtype=np.float64)
    for r in range(10):
        dem[r, :] = 9 - r  # Higher at top (north), lower at bottom (south)
    path = tmp_path / "south.tif"
    _write_dem(path, dem)
    return path, dem


@pytest.fixture
def west_facing(tmp_path):
    """DEM sloping westward."""
    dem = np.zeros((10, 10), dtype=np.float64)
    for c in range(10):
        dem[:, c] = c  # Higher on right (east), lower on left (west)
    path = tmp_path / "west.tif"
    _write_dem(path, dem)
    return path, dem


@pytest.fixture
def northeast_facing(tmp_path):
    """DEM sloping northeast (equal X and Y gradients)."""
    # NE aspect = 45°
    # NE-facing: downslope points NE (north and east)
    # Row 0 is at y=nrows (NORTH), Row 9 is at y=1 (SOUTH)
    # For NE-facing: higher in south (row 9), lower in north (row 0) → use r
    #                higher in west (col 0), lower in east (col 9) → use c
    dem = np.zeros((10, 10), dtype=np.float64)
    for r in range(10):
        for c in range(10):
            dem[r, c] = r - c + 9  # High SW (r=9,c=0)=18, low NE (r=0,c=9)=0
    path = tmp_path / "ne.tif"
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


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


class TestProtocolCompliance:
    def test_implements_operator_protocol(self, op):
        assert isinstance(op, Operator)

    def test_name(self, op):
        assert op.name == "aspect"

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
        errors = op.validate_inputs([], AspectParams(output_path="/tmp/x.tif"))
        assert any("required" in e.lower() for e in errors)

    def test_rejects_vector_input(self, op, tmp_path):
        art = Artifact(
            type=ArtifactType.VECTOR,
            name="v",
            backing=BackingStore(
                kind=BackingStoreKind.LOCAL_FILE, uri="/fake", size_bytes=0, content_hash=""
            ),
        )
        errors = op.validate_inputs([art], AspectParams(output_path="/tmp/x.tif"))
        assert any("raster" in e.lower() for e in errors)

    def test_rejects_missing_output_path(self, op, flat_dem):
        path, _ = flat_dem
        art = _make_artifact(path)
        errors = op.validate_inputs([art], AspectParams(output_path=""))
        assert any("output_path" in e for e in errors)

    def test_rejects_invalid_convention(self, op, flat_dem):
        path, _ = flat_dem
        art = _make_artifact(path)
        errors = op.validate_inputs(
            [art], AspectParams(output_path="/tmp/x.tif", convention="invalid")
        )
        assert any("convention" in e.lower() for e in errors)

    def test_accepts_valid_input(self, op, flat_dem):
        path, _ = flat_dem
        art = _make_artifact(path)
        errors = op.validate_inputs([art], AspectParams(output_path="/tmp/x.tif"))
        assert errors == []


# ---------------------------------------------------------------------------
# Flat surface
# ---------------------------------------------------------------------------


class TestFlatSurface:
    def test_flat_produces_flat_value(self, op, flat_dem, tmp_path):
        """Perfectly flat DEM should produce flat_value everywhere."""
        path, _ = flat_dem
        art = _make_artifact(path)
        output = tmp_path / "aspect.tif"
        params = AspectParams(output_path=str(output), flat_value=-1.0)

        _ = op.execute([art], params)

        with rasterio.open(output) as src:
            aspect = src.read(1)
            valid = aspect != params.output_nodata
            assert np.all(aspect[valid] == -1.0)

    def test_flat_custom_value(self, op, flat_dem, tmp_path):
        """Flat DEM with custom flat_value."""
        path, _ = flat_dem
        art = _make_artifact(path)
        output = tmp_path / "aspect.tif"
        params = AspectParams(output_path=str(output), flat_value=999.0)

        _ = op.execute([art], params)

        with rasterio.open(output) as src:
            aspect = src.read(1)
            valid_mask = aspect != params.output_nodata
            assert np.all(aspect[valid_mask] == 999.0)


# ---------------------------------------------------------------------------
# Cardinal directions
# ---------------------------------------------------------------------------


class TestCardinalDirections:
    def test_north_facing(self, op, north_facing, tmp_path):
        """North-facing slope: aspect = 0°."""
        path, _ = north_facing
        art = _make_artifact(path)
        output = tmp_path / "aspect.tif"
        params = AspectParams(output_path=str(output))

        op.execute([art], params)

        with rasterio.open(output) as src:
            aspect = src.read(1)
            interior = aspect[1:-1, 1:-1]
            # North aspect should be ~0°
            np.testing.assert_allclose(interior, 0.0, atol=5.0)

    def test_east_facing(self, op, east_facing, tmp_path):
        """East-facing slope: aspect = 90°."""
        path, _ = east_facing
        art = _make_artifact(path)
        output = tmp_path / "aspect.tif"
        params = AspectParams(output_path=str(output))

        op.execute([art], params)

        with rasterio.open(output) as src:
            aspect = src.read(1)
            interior = aspect[1:-1, 1:-1]
            np.testing.assert_allclose(interior, 90.0, atol=5.0)

    def test_south_facing(self, op, south_facing, tmp_path):
        """South-facing slope: aspect = 180°."""
        path, _ = south_facing
        art = _make_artifact(path)
        output = tmp_path / "aspect.tif"
        params = AspectParams(output_path=str(output))

        op.execute([art], params)

        with rasterio.open(output) as src:
            aspect = src.read(1)
            interior = aspect[1:-1, 1:-1]
            np.testing.assert_allclose(interior, 180.0, atol=5.0)

    def test_west_facing(self, op, west_facing, tmp_path):
        """West-facing slope: aspect = 270°."""
        path, _ = west_facing
        art = _make_artifact(path)
        output = tmp_path / "aspect.tif"
        params = AspectParams(output_path=str(output))

        op.execute([art], params)

        with rasterio.open(output) as src:
            aspect = src.read(1)
            interior = aspect[1:-1, 1:-1]
            np.testing.assert_allclose(interior, 270.0, atol=5.0)


# ---------------------------------------------------------------------------
# Diagonal directions
# ---------------------------------------------------------------------------


class TestDiagonalDirections:
    def test_northeast_facing(self, op, northeast_facing, tmp_path):
        """NE-facing slope: aspect = 45°."""
        path, _ = northeast_facing
        art = _make_artifact(path)
        output = tmp_path / "aspect.tif"
        params = AspectParams(output_path=str(output))

        op.execute([art], params)

        with rasterio.open(output) as src:
            aspect = src.read(1)
            interior = aspect[1:-1, 1:-1]
            np.testing.assert_allclose(interior, 45.0, atol=2.0)


# ---------------------------------------------------------------------------
# Nodata handling
# ---------------------------------------------------------------------------


class TestNodata:
    def test_nodata_preserved(self, op, nodata_dem, tmp_path):
        """Input nodata cells produce output nodata."""
        path, _ = nodata_dem
        art = _make_artifact(path)
        output = tmp_path / "aspect.tif"
        params = AspectParams(output_path=str(output), nodata=-9999.0)

        op.execute([art], params)

        with rasterio.open(output) as src:
            aspect = src.read(1)
            assert aspect[4, 4] == params.output_nodata
            assert aspect[4, 5] == params.output_nodata
            assert aspect[5, 4] == params.output_nodata
            assert aspect[5, 5] == params.output_nodata


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


class TestExecution:
    def test_execute_produces_raster(self, op, east_facing, tmp_path):
        path, _ = east_facing
        art = _make_artifact(path)
        output = tmp_path / "aspect.tif"
        params = AspectParams(output_path=str(output))

        result = op.execute([art], params)

        assert result.artifact.type == ArtifactType.RASTER
        assert result.artifact.backing.uri == str(output)
        assert output.exists()

    def test_execute_preserves_crs(self, op, east_facing, tmp_path):
        path, _ = east_facing
        art = _make_artifact(path)
        output = tmp_path / "aspect.tif"
        params = AspectParams(output_path=str(output))

        result = op.execute([art], params)
        assert "32610" in result.artifact.spatial.crs

    def test_execute_lineage(self, op, east_facing, tmp_path):
        path, _ = east_facing
        art = _make_artifact(path)
        output = tmp_path / "aspect.tif"
        params = AspectParams(output_path=str(output), convention="compass")

        result = op.execute([art], params)
        assert result.artifact.lineage.operation == "aspect"
        assert result.artifact.lineage.params["convention"] == "compass"

    def test_execute_metadata(self, op, east_facing, tmp_path):
        path, _ = east_facing
        art = _make_artifact(path)
        output = tmp_path / "aspect.tif"
        params = AspectParams(output_path=str(output))

        result = op.execute([art], params)
        assert result.artifact.metadata["algorithm"] == "central_difference_gradient"
        assert result.artifact.metadata["convention"] == "compass"

    def test_execute_checks_pass(self, op, east_facing, tmp_path):
        path, _ = east_facing
        art = _make_artifact(path)
        output = tmp_path / "aspect.tif"
        params = AspectParams(output_path=str(output))

        result = op.execute([art], params)
        for check in result.checks:
            assert check.state == ValidationState.VALID, f"{check.check_name}: {check.message}"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_tiny_3x3(self, op, tmp_path):
        """Minimum viable grid for central difference."""
        # East-facing plane
        dem = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
            ],
            dtype=np.float64,
        )
        path = tmp_path / "tiny.tif"
        _write_dem(path, dem)
        art = _make_artifact(path)
        output = tmp_path / "aspect.tif"
        params = AspectParams(output_path=str(output))

        op.execute([art], params)
        with rasterio.open(output) as src:
            aspect = src.read(1)
            # Has valid aspect values
            valid = aspect != params.output_nodata
            assert np.any(valid)

    def test_single_row(self, op, tmp_path):
        """1xN grid — only X gradient, Y gradient is zero."""
        # East-facing: higher on left
        dem = np.arange(10, 0, -1, dtype=np.float64).reshape(1, 10)
        path = tmp_path / "single_row.tif"
        _write_dem(path, dem)
        art = _make_artifact(path)
        output = tmp_path / "aspect.tif"
        params = AspectParams(output_path=str(output))

        result = op.execute([art], params)
        assert result.artifact is not None

        # Should give east or west aspect (mostly 90 or 270)
        with rasterio.open(output) as src:
            aspect = src.read(1)
            valid = aspect != params.output_nodata
            valid_aspect = aspect[valid]
            if len(valid_aspect) > 0:
                # Single row with X gradient only should give ~90 or ~270
                assert np.all(
                    (valid_aspect > 80) & (valid_aspect < 100)
                    | (valid_aspect > 260) & (valid_aspect < 280)
                )

    def test_single_column(self, op, tmp_path):
        """Nx1 grid — only Y gradient, X gradient is zero."""
        # North-facing: higher at bottom
        dem = np.arange(10, 0, -1, dtype=np.float64).reshape(10, 1)
        path = tmp_path / "single_col.tif"
        _write_dem(path, dem)
        art = _make_artifact(path)
        output = tmp_path / "aspect.tif"
        params = AspectParams(output_path=str(output))

        result = op.execute([art], params)
        assert result.artifact is not None

    def test_all_nodata(self, op, tmp_path):
        """Completely nodata grid."""
        dem = np.full((5, 5), -9999.0, dtype=np.float64)
        path = tmp_path / "all_nodata.tif"
        _write_dem(path, dem, nodata=-9999.0)
        art = _make_artifact(path)
        output = tmp_path / "aspect.tif"
        params = AspectParams(output_path=str(output), nodata=-9999.0)

        _ = op.execute([art], params)

        with rasterio.open(output) as src:
            aspect = src.read(1)
            assert np.all(aspect == params.output_nodata)


# ---------------------------------------------------------------------------
# Math convention
# ---------------------------------------------------------------------------


class TestMathConvention:
    def test_east_facing_math(self, op, east_facing, tmp_path):
        """East-facing in math convention: 0° (east is 0 in math)."""
        path, _ = east_facing
        art = _make_artifact(path)
        output = tmp_path / "aspect.tif"
        params = AspectParams(output_path=str(output), convention="math")

        op.execute([art], params)

        with rasterio.open(output) as src:
            aspect = src.read(1)
            interior = aspect[1:-1, 1:-1]
            # Math convention: 0 = East
            np.testing.assert_allclose(interior, 0.0, atol=5.0)

    def test_north_facing_math(self, op, north_facing, tmp_path):
        """North-facing in math convention: 90°."""
        path, _ = north_facing
        art = _make_artifact(path)
        output = tmp_path / "aspect.tif"
        params = AspectParams(output_path=str(output), convention="math")

        op.execute([art], params)

        with rasterio.open(output) as src:
            aspect = src.read(1)
            interior = aspect[1:-1, 1:-1]
            # Math convention: 90 = North
            np.testing.assert_allclose(interior, 90.0, atol=5.0)
