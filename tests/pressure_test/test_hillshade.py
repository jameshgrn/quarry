"""Hillshade operator pressure test.

Lane: operator

Stress points:
1. Flat DEM produces uniform illumination (cos(zenith)*255 for uint8)
2. Slope facing sun produces maximum illumination (~255 or 1.0)
3. Slope facing away from sun produces minimum illumination (~0)
4. Sun directly overhead (altitude=90°) produces illumination = cos(slope)*255
5. Nodata cells produce nodata in output
6. Parameter variations (azimuth, altitude, z_factor) produce expected changes
7. Output format: uint8 (0-255) default, float64 (0.0-1.0) when scaled=True
8. Horn (1981) algorithm correctness on synthetic surfaces
9. Operator protocol compliance (spec, validate_inputs, declared_checks)
10. Lineage records algorithm params (azimuth, altitude, z_factor, scaled)
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
from quarry_core.operator import Operator, OperatorError, ResourceScale
from quarry_operators.hillshade import HillshadeOperator, HillshadeParams
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
    return HillshadeOperator()


@pytest.fixture
def flat_dem(tmp_path):
    """Perfectly flat 10x10 DEM at elevation 100."""
    dem = np.full((10, 10), 100.0, dtype=np.float64)
    path = tmp_path / "flat.tif"
    _write_dem(path, dem)
    return path, dem


@pytest.fixture
def east_facing_slope(tmp_path):
    """DEM with east-facing slope (surface faces east, elevation decreases eastward).

    With sun at azimuth=90° (east), this slope faces the sun → bright.
    East-facing means the downslope direction is east, so elevation decreases eastward.
    """
    dem = np.zeros((10, 10), dtype=np.float64)
    for c in range(10):
        dem[:, c] = 9 - c  # Elevation decreases eastward (high west, low east)
    path = tmp_path / "east_slope.tif"
    _write_dem(path, dem)
    return path, dem


@pytest.fixture
def west_facing_slope(tmp_path):
    """DEM with west-facing slope (surface faces west, elevation decreases westward).

    With sun at azimuth=90° (east), this slope faces away from the sun → dark.
    West-facing means the downslope direction is west, so elevation decreases westward.
    """
    dem = np.zeros((10, 10), dtype=np.float64)
    for c in range(10):
        dem[:, c] = c  # Elevation decreases westward (high east, low west)
    path = tmp_path / "west_slope.tif"
    _write_dem(path, dem)
    return path, dem


@pytest.fixture
def north_facing_slope(tmp_path):
    """DEM with north-facing slope (surface faces north, elevation decreases northward).

    Row 0 is at y=10 (north), row 9 is at y=1 (south) per from_origin.
    Elevation increases with row index → high in south, low in north → downslope is north.
    """
    dem = np.zeros((10, 10), dtype=np.float64)
    for r in range(10):
        dem[r, :] = r  # Row 0 (north) = 0, Row 9 (south) = 9
    path = tmp_path / "north_slope.tif"
    _write_dem(path, dem)
    return path, dem


@pytest.fixture
def south_facing_slope(tmp_path):
    """DEM with south-facing slope (surface faces south, elevation decreases southward).

    Row 0 is at y=10 (north), row 9 is at y=1 (south) per from_origin.
    Elevation decreases with row index → high in north, low in south → downslope is south.
    """
    dem = np.zeros((10, 10), dtype=np.float64)
    for r in range(10):
        dem[r, :] = 9 - r  # Row 0 (north) = 9, Row 9 (south) = 0
    path = tmp_path / "south_slope.tif"
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
def inclined_plane_45(tmp_path):
    """DEM inclined at exactly 45 degrees (1:1 rise:run)."""
    dem = np.zeros((10, 10), dtype=np.float64)
    for c in range(10):
        dem[:, c] = c  # Elevation = column index
    path = tmp_path / "inclined_45.tif"
    _write_dem(path, dem)
    return path, dem


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


class TestProtocolCompliance:
    def test_implements_operator_protocol(self, op):
        assert isinstance(op, Operator)

    def test_name(self, op):
        assert op.name == "hillshade"

    def test_spec(self, op):
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
    def test_validate_no_inputs(self, op):
        errors = op.validate_inputs([], HillshadeParams(output_path="/tmp/x.tif"))
        assert any("required" in e.lower() for e in errors)

    def test_validate_too_many_inputs(self, op, flat_dem):
        path, _ = flat_dem
        art = _make_artifact(path)
        errors = op.validate_inputs([art, art], HillshadeParams(output_path="/tmp/x.tif"))
        assert any("expected 1" in e.lower() for e in errors)

    def test_validate_wrong_type(self, op, tmp_path):
        art = Artifact(
            type=ArtifactType.VECTOR,
            name="v",
            backing=BackingStore(
                kind=BackingStoreKind.LOCAL_FILE, uri="/fake", size_bytes=0, content_hash=""
            ),
        )
        errors = op.validate_inputs([art], HillshadeParams(output_path="/tmp/x.tif"))
        assert any("raster" in e.lower() for e in errors)

    def test_validate_lazy_artifact(self, op, flat_dem):
        path, _ = flat_dem
        art = _make_artifact(path)
        # Create a lazy artifact by using LAZY_HANDLE backing store
        lazy_art = Artifact(
            type=art.type,
            name=art.name,
            backing=BackingStore(
                kind=BackingStoreKind.LAZY_HANDLE,  # Not LOCAL_FILE
                uri=art.backing.uri,
                size_bytes=art.backing.size_bytes,
                content_hash=art.backing.content_hash,
            ),
            spatial=art.spatial,
        )
        errors = op.validate_inputs([lazy_art], HillshadeParams(output_path="/tmp/x.tif"))
        assert any("materialized" in e.lower() for e in errors)

    def test_validate_no_output_path(self, op, flat_dem):
        path, _ = flat_dem
        art = _make_artifact(path)
        errors = op.validate_inputs([art], HillshadeParams(output_path=""))
        assert any("output_path" in e for e in errors)

    @pytest.mark.parametrize("azimuth", [-1.0, 361.0, 400.0])
    def test_validate_invalid_azimuth(self, op, flat_dem, azimuth):
        path, _ = flat_dem
        art = _make_artifact(path)
        errors = op.validate_inputs(
            [art], HillshadeParams(output_path="/tmp/x.tif", azimuth=azimuth)
        )
        assert any("azimuth" in e.lower() for e in errors)

    @pytest.mark.parametrize("altitude", [-1.0, 91.0, 100.0])
    def test_validate_invalid_altitude(self, op, flat_dem, altitude):
        path, _ = flat_dem
        art = _make_artifact(path)
        errors = op.validate_inputs(
            [art], HillshadeParams(output_path="/tmp/x.tif", altitude=altitude)
        )
        assert any("altitude" in e.lower() for e in errors)

    @pytest.mark.parametrize("z_factor", [0.0, -1.0, -0.5])
    def test_validate_invalid_z_factor(self, op, flat_dem, z_factor):
        path, _ = flat_dem
        art = _make_artifact(path)
        errors = op.validate_inputs(
            [art], HillshadeParams(output_path="/tmp/x.tif", z_factor=z_factor)
        )
        assert any("z_factor" in e.lower() for e in errors)

    def test_validate_azimuth_360_boundary(self, op, flat_dem):
        """azimuth=360 is valid (equivalent to 0=North)."""
        path, _ = flat_dem
        art = _make_artifact(path)
        errors = op.validate_inputs([art], HillshadeParams(output_path="/tmp/x.tif", azimuth=360.0))
        assert errors == []

    def test_validate_invalid_output_nodata_uint8(self, op, flat_dem):
        """output_nodata outside 0-255 range rejected for uint8 output."""
        path, _ = flat_dem
        art = _make_artifact(path)
        errors = op.validate_inputs(
            [art], HillshadeParams(output_path="/tmp/x.tif", output_nodata=256.0)
        )
        assert any("output_nodata" in e for e in errors)

    def test_validate_noninteger_output_nodata_uint8(self, op, flat_dem):
        """Non-integer output_nodata rejected for uint8 output."""
        path, _ = flat_dem
        art = _make_artifact(path)
        errors = op.validate_inputs(
            [art], HillshadeParams(output_path="/tmp/x.tif", output_nodata=127.5)
        )
        assert any("integer" in e.lower() for e in errors)

    def test_accepts_valid_input(self, op, flat_dem):
        path, _ = flat_dem
        art = _make_artifact(path)
        errors = op.validate_inputs([art], HillshadeParams(output_path="/tmp/x.tif"))
        assert errors == []


# ---------------------------------------------------------------------------
# Correctness — known surfaces
# ---------------------------------------------------------------------------


class TestFlatSurface:
    def test_flat_dem(self, op, flat_dem, tmp_path):
        """Flat DEM with default altitude=45° should produce uniform illumination.

        With altitude=45°, zenith=45°, so cos(zenith)=cos(45°)=~0.707.
        Expected uint8 value: 0.707 * 255 ≈ 180.
        """
        path, _ = flat_dem
        art = _make_artifact(path)
        output = tmp_path / "hillshade.tif"
        params = HillshadeParams(output_path=str(output), altitude=45.0)

        op.execute([art], params)

        with rasterio.open(output) as src:
            hillshade = src.read(1)
            valid = hillshade != params.output_nodata
            # For flat surface with altitude=45°, expected = cos(45°) * 255 ≈ 180
            expected = int(math.cos(math.radians(45.0)) * 255)
            # All valid cells should have approximately the same value
            unique_values = np.unique(hillshade[valid])
            assert len(unique_values) == 1
            assert abs(unique_values[0] - expected) <= 1


class TestSlopedSurfaces:
    def test_45_degree_slope_facing_sun(self, op, east_facing_slope, tmp_path):
        """45° slope facing sun at azimuth=90° should produce near-maximum illumination."""
        path, _ = east_facing_slope
        art = _make_artifact(path)
        output = tmp_path / "hillshade.tif"
        # Sun from east (90°), slope faces east
        params = HillshadeParams(output_path=str(output), azimuth=90.0, altitude=45.0)

        op.execute([art], params)

        with rasterio.open(output) as src:
            hillshade = src.read(1)
            interior = hillshade[1:-1, 1:-1]
            valid = interior != params.output_nodata
            # Should be near maximum (255) since slope faces sun
            # Horn formula: cos(45°)*cos(45°) + sin(45°)*sin(45°)*cos(0°) = 0.5 + 0.5 = 1.0 → 255
            assert np.all(interior[valid] > 240)

    def test_45_degree_slope_facing_away(self, op, west_facing_slope, tmp_path):
        """45° slope facing away from sun should produce near-minimum illumination."""
        path, _ = west_facing_slope
        art = _make_artifact(path)
        output = tmp_path / "hillshade.tif"
        # Sun from east (90°), slope faces west (away from sun)
        params = HillshadeParams(output_path=str(output), azimuth=90.0, altitude=45.0)

        op.execute([art], params)

        with rasterio.open(output) as src:
            hillshade = src.read(1)
            interior = hillshade[1:-1, 1:-1]
            valid = interior != params.output_nodata
            # Should be near minimum (0) since slope faces away from sun
            # Horn formula: cos(45°)*cos(45°) + sin(45°)*sin(45°)*cos(180°) = 0.5 - 0.5 = 0.0 → 0
            assert np.all(interior[valid] < 20)

    def test_sun_directly_overhead(self, op, inclined_plane_45, tmp_path):
        """Sun at altitude=90° (zenith=0°) → illumination = cos(slope)*255 regardless of aspect."""
        path, _ = inclined_plane_45
        art = _make_artifact(path)
        output = tmp_path / "hillshade.tif"
        # Sun directly overhead
        params = HillshadeParams(output_path=str(output), altitude=90.0)

        op.execute([art], params)

        with rasterio.open(output) as src:
            hillshade = src.read(1)
            interior = hillshade[1:-1, 1:-1]
            valid = interior != params.output_nodata
            # With altitude=90°, zenith=0°, cos(zenith)=1
            # illumination = cos(slope) * 255
            # For 45° slope, cos(45°) * 255 ≈ 180
            expected = int(math.cos(math.radians(45.0)) * 255)
            # Values should be consistent (aspect doesn't matter when sun is overhead)
            mean_val = np.mean(interior[valid])
            assert abs(mean_val - expected) < 10

    def test_sun_at_horizon(self, op, east_facing_slope, tmp_path):
        """Sun at altitude=0° (horizon) → illumination depends purely on aspect alignment."""
        path, _ = east_facing_slope
        art = _make_artifact(path)
        output = tmp_path / "hillshade.tif"
        # Sun at horizon from east
        params = HillshadeParams(output_path=str(output), azimuth=90.0, altitude=0.0)

        op.execute([art], params)

        with rasterio.open(output) as src:
            hillshade = src.read(1)
            interior = hillshade[1:-1, 1:-1]
            valid = interior != params.output_nodata
            # With sun at horizon (altitude=0°): zenith=90°, cos(z)=0, sin(z)=1
            # illumination = sin(slope) * cos(azimuth - aspect)
            # For 45° slope facing sun directly: sin(45°)*cos(0) ≈ 0.707 → uint8 ≈ 180
            assert np.all(interior[valid] > 170)

    def test_east_facing_slope_east_sun(self, op, east_facing_slope, tmp_path):
        """East-facing slope with sun from east → high illumination."""
        path, _ = east_facing_slope
        art = _make_artifact(path)
        output = tmp_path / "hillshade.tif"
        params = HillshadeParams(output_path=str(output), azimuth=90.0, altitude=45.0)

        op.execute([art], params)

        with rasterio.open(output) as src:
            hillshade = src.read(1)
            interior = hillshade[1:-1, 1:-1]
            valid = interior != params.output_nodata
            # Horn formula: cos(45°)*cos(45°) + sin(45°)*sin(45°)*cos(0°) = 1.0 → 255
            assert np.all(interior[valid] > 240)

    def test_north_facing_slope_south_sun(self, op, north_facing_slope, tmp_path):
        """North-facing slope with sun from south (azimuth=180°) → low illumination."""
        path, _ = north_facing_slope
        art = _make_artifact(path)
        output = tmp_path / "hillshade.tif"
        # Sun from south, slope faces north (away from sun)
        params = HillshadeParams(output_path=str(output), azimuth=180.0, altitude=45.0)

        op.execute([art], params)

        with rasterio.open(output) as src:
            hillshade = src.read(1)
            interior = hillshade[1:-1, 1:-1]
            valid = interior != params.output_nodata
            # Should be relatively dark since slope faces away from sun
            # Horn formula: cos(45°)*cos(45°) + sin(45°)*sin(45°)*cos(180°) = 0.5 - 0.5 = 0.0 → 0
            assert np.all(interior[valid] < 20)

    def test_south_facing_slope_south_sun(self, op, south_facing_slope, tmp_path):
        """South-facing slope with sun from south (azimuth=180°) → high illumination."""
        path, _ = south_facing_slope
        art = _make_artifact(path)
        output = tmp_path / "hillshade.tif"
        # Sun from south, slope faces south (toward sun)
        params = HillshadeParams(output_path=str(output), azimuth=180.0, altitude=45.0)

        op.execute([art], params)

        with rasterio.open(output) as src:
            hillshade = src.read(1)
            interior = hillshade[1:-1, 1:-1]
            valid = interior != params.output_nodata
            # Should be bright since slope faces the sun
            # Horn formula: cos(45°)*cos(45°) + sin(45°)*sin(45°)*cos(0°) = 0.5 + 0.5 = 1.0 → 255
            assert np.all(interior[valid] > 240)


# ---------------------------------------------------------------------------
# Output format
# ---------------------------------------------------------------------------


class TestOutputFormat:
    def test_default_uint8_output(self, op, flat_dem, tmp_path):
        """Default output is uint8 with values 0-255."""
        path, _ = flat_dem
        art = _make_artifact(path)
        output = tmp_path / "hillshade.tif"
        params = HillshadeParams(output_path=str(output))

        result = op.execute([art], params)

        with rasterio.open(output) as src:
            assert src.dtypes[0] == "uint8"
            hillshade = src.read(1)
            valid = hillshade != params.output_nodata
            assert np.all(hillshade[valid] >= 0)
            assert np.all(hillshade[valid] <= 255)
        assert result.artifact.metadata["dtype"] == "uint8"

    def test_scaled_float_output(self, op, flat_dem, tmp_path):
        """scaled=True produces float64 output with values 0.0-1.0."""
        path, _ = flat_dem
        art = _make_artifact(path)
        output = tmp_path / "hillshade.tif"
        params = HillshadeParams(output_path=str(output), scaled=True)

        result = op.execute([art], params)

        with rasterio.open(output) as src:
            assert src.dtypes[0] == "float64"
            hillshade = src.read(1)
            valid = hillshade != params.output_nodata
            assert np.all(hillshade[valid] >= 0.0)
            assert np.all(hillshade[valid] <= 1.0)
        assert result.artifact.metadata["dtype"] == "float64"

    def test_output_range_clamped(self, op, east_facing_slope, tmp_path):
        """Illumination values are clamped to valid range."""
        path, _ = east_facing_slope
        art = _make_artifact(path)

        # Test uint8 output
        output1 = tmp_path / "hillshade_uint8.tif"
        params1 = HillshadeParams(output_path=str(output1))
        op.execute([art], params1)

        with rasterio.open(output1) as src:
            hillshade = src.read(1)
            valid = hillshade != params1.output_nodata
            assert np.all(hillshade[valid] >= 0)
            assert np.all(hillshade[valid] <= 255)

        # Test float64 output
        output2 = tmp_path / "hillshade_float.tif"
        params2 = HillshadeParams(output_path=str(output2), scaled=True)
        op.execute([art], params2)

        with rasterio.open(output2) as src:
            hillshade = src.read(1)
            valid = hillshade != params2.output_nodata
            assert np.all(hillshade[valid] >= 0.0)
            assert np.all(hillshade[valid] <= 1.0)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_row(self, op, tmp_path):
        """1×N grid should not crash."""
        dem = np.arange(10, dtype=np.float64).reshape(1, 10)
        path = tmp_path / "single_row.tif"
        _write_dem(path, dem)
        art = _make_artifact(path)
        output = tmp_path / "hillshade.tif"
        params = HillshadeParams(output_path=str(output))

        result = op.execute([art], params)
        assert result.artifact is not None

        with rasterio.open(output) as src:
            hillshade = src.read(1)
            assert hillshade.shape == (1, 10)

    def test_single_column(self, op, tmp_path):
        """N×1 grid should not crash."""
        dem = np.arange(10, dtype=np.float64).reshape(10, 1)
        path = tmp_path / "single_col.tif"
        _write_dem(path, dem)
        art = _make_artifact(path)
        output = tmp_path / "hillshade.tif"
        params = HillshadeParams(output_path=str(output))

        result = op.execute([art], params)
        assert result.artifact is not None

        with rasterio.open(output) as src:
            hillshade = src.read(1)
            assert hillshade.shape == (10, 1)

    def test_all_nodata(self, op, tmp_path):
        """Completely nodata grid should produce all-nodata output."""
        dem = np.full((5, 5), -9999.0, dtype=np.float64)
        path = tmp_path / "all_nodata.tif"
        _write_dem(path, dem, nodata=-9999.0)
        art = _make_artifact(path)
        output = tmp_path / "hillshade.tif"
        params = HillshadeParams(output_path=str(output), nodata=-9999.0)

        op.execute([art], params)

        with rasterio.open(output) as src:
            hillshade = src.read(1)
            assert np.all(hillshade == params.output_nodata)

    def test_tiny_3x3(self, op, tmp_path):
        """3×3 DEM with known values should produce correct output."""
        # East-facing plane (elevation decreases eastward)
        dem = np.array(
            [
                [2.0, 1.0, 0.0],
                [2.0, 1.0, 0.0],
                [2.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        path = tmp_path / "tiny.tif"
        _write_dem(path, dem)
        art = _make_artifact(path)
        output = tmp_path / "hillshade.tif"
        params = HillshadeParams(output_path=str(output), azimuth=90.0, altitude=45.0)

        op.execute([art], params)

        with rasterio.open(output) as src:
            hillshade = src.read(1)
            # Center cell should have valid value
            assert hillshade[1, 1] != params.output_nodata
            # East-facing plane with sun from east at 45°: expect bright center
            assert hillshade[1, 1] > 240

    def test_multiband_rejected(self, op, tmp_path):
        """Multi-band raster should raise OperatorError."""
        # Create a multi-band raster
        dem = np.full((10, 10), 100.0, dtype=np.float64)
        path = tmp_path / "multiband.tif"

        meta = {
            "driver": "GTiff",
            "height": 10,
            "width": 10,
            "count": 3,  # Multi-band
            "dtype": "float64",
            "crs": CRS.from_epsg(32610),
            "transform": from_origin(0, 10, 1, 1),
        }
        with rasterio.open(path, "w", **meta) as dst:
            for i in range(3):
                dst.write(dem, i + 1)

        art = _make_artifact(path)
        output = tmp_path / "hillshade.tif"
        params = HillshadeParams(output_path=str(output))

        with pytest.raises(OperatorError) as exc_info:
            op.execute([art], params)
        assert "single-band" in str(exc_info.value).lower() or "band" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# Parameter variations
# ---------------------------------------------------------------------------


class TestParameterVariations:
    def test_z_factor_scaling(self, op, east_facing_slope, tmp_path):
        """z_factor=2.0 should produce different (more contrast) result than z_factor=1.0."""
        path, _ = east_facing_slope
        art = _make_artifact(path)

        output1 = tmp_path / "hillshade_z1.tif"
        params1 = HillshadeParams(output_path=str(output1), z_factor=1.0, azimuth=90.0)
        op.execute([art], params1)

        output2 = tmp_path / "hillshade_z2.tif"
        params2 = HillshadeParams(output_path=str(output2), z_factor=2.0, azimuth=90.0)
        op.execute([art], params2)

        with rasterio.open(output1) as src1, rasterio.open(output2) as src2:
            h1 = src1.read(1)
            h2 = src2.read(1)
            valid1 = h1 != params1.output_nodata
            valid2 = h2 != params2.output_nodata
            # Results should be different
            assert not np.allclose(h1[valid1].astype(float), h2[valid2].astype(float))

    @pytest.mark.parametrize("azimuth", [0.0, 90.0, 180.0, 270.0])
    def test_custom_azimuth(self, op, east_facing_slope, tmp_path, azimuth):
        """Different azimuths produce measurably different illumination on sloped surface."""
        path, _ = east_facing_slope
        art = _make_artifact(path)
        output = tmp_path / f"hillshade_az{azimuth}.tif"
        params = HillshadeParams(output_path=str(output), azimuth=azimuth, altitude=45.0)

        result = op.execute([art], params)
        assert result.artifact is not None

        with rasterio.open(output) as src:
            hillshade = src.read(1)
            interior = hillshade[1:-1, 1:-1]
            # No actual nodata in this DEM, so compute mean over all interior pixels
            mean_val = float(np.mean(interior))
            # East-facing slope: azimuth=90 (sun from east) should be brightest
            if azimuth == 90.0:
                assert mean_val > 240
            # azimuth=270 (sun from west, opposite to slope) should be darkest
            elif azimuth == 270.0:
                assert mean_val < 20

    @pytest.mark.parametrize("altitude", [30.0, 45.0, 60.0])
    def test_custom_altitude(self, op, flat_dem, tmp_path, altitude):
        """Different altitudes should produce different results."""
        path, _ = flat_dem
        art = _make_artifact(path)
        output = tmp_path / f"hillshade_alt{altitude}.tif"
        params = HillshadeParams(output_path=str(output), altitude=altitude)

        op.execute([art], params)

        with rasterio.open(output) as src:
            hillshade = src.read(1)
            valid = hillshade != params.output_nodata
            # Higher altitude = more direct light = brighter
            mean_val = np.mean(hillshade[valid])
            # altitude=60° should be brighter than altitude=30°
            expected_60 = int(math.cos(math.radians(30.0)) * 255)  # zenith = 30°
            expected_30 = int(math.cos(math.radians(60.0)) * 255)  # zenith = 60°
            if altitude == 60.0:
                assert abs(mean_val - expected_60) <= 1
            elif altitude == 30.0:
                assert abs(mean_val - expected_30) <= 1


# ---------------------------------------------------------------------------
# Nodata handling
# ---------------------------------------------------------------------------


class TestNodataHandling:
    def test_nodata_preserved(self, op, nodata_dem, tmp_path):
        """Input nodata cells should produce output nodata cells.

        Note: Hillshade gradient calculation may expand nodata region slightly
        because edge cells adjacent to nodata cannot compute valid gradients.
        """
        path, _ = nodata_dem
        art = _make_artifact(path)
        output = tmp_path / "hillshade.tif"
        params = HillshadeParams(output_path=str(output), nodata=-9999.0)

        result = op.execute([art], params)

        with rasterio.open(output) as src:
            hillshade = src.read(1)
            # The original 2x2 nodata region should be nodata in output
            assert hillshade[4, 4] == params.output_nodata
            assert hillshade[4, 5] == params.output_nodata
            assert hillshade[5, 4] == params.output_nodata
            assert hillshade[5, 5] == params.output_nodata

        # Check that nodata_preserved check exists (may be WARN due to gradient expansion)
        nodata_check = next(c for c in result.checks if c.check_name == "nodata_preserved")
        assert nodata_check.state in (ValidationState.VALID, ValidationState.WARN)

    def test_nan_nodata(self, op, tmp_path):
        """NaN values should be handled as nodata."""
        dem = np.full((5, 5), 100.0, dtype=np.float64)
        dem[2, 2] = np.nan
        path = tmp_path / "nan_nodata.tif"
        _write_dem(path, dem)
        art = _make_artifact(path)
        output = tmp_path / "hillshade.tif"
        params = HillshadeParams(output_path=str(output))

        result = op.execute([art], params)
        assert result.artifact is not None


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


class TestChecks:
    def test_all_checks_pass_happy_path(self, op, east_facing_slope, tmp_path):
        """All 3 checks should be valid on good input."""
        path, _ = east_facing_slope
        art = _make_artifact(path)
        output = tmp_path / "hillshade.tif"
        params = HillshadeParams(output_path=str(output))

        result = op.execute([art], params)

        for check in result.checks:
            assert check.state == ValidationState.VALID, f"{check.check_name}: {check.message}"


# ---------------------------------------------------------------------------
# Artifact/lineage
# ---------------------------------------------------------------------------


class TestArtifactAndLineage:
    def test_fresh_metadata(self, op, flat_dem, tmp_path):
        """Output metadata should come from actual file, not copied from input."""
        path, _ = flat_dem
        art = _make_artifact(path)
        output = tmp_path / "hillshade.tif"
        params = HillshadeParams(output_path=str(output))

        result = op.execute([art], params)

        # Verify metadata reflects actual output file
        assert result.artifact.metadata["algorithm"] == "horn_1981"
        assert "azimuth" in result.artifact.metadata
        assert "altitude" in result.artifact.metadata
        assert "z_factor" in result.artifact.metadata

    def test_lineage_records_params(self, op, flat_dem, tmp_path):
        """Lineage should contain azimuth, altitude, z_factor, scaled."""
        path, _ = flat_dem
        art = _make_artifact(path)
        output = tmp_path / "hillshade.tif"
        params = HillshadeParams(
            output_path=str(output),
            azimuth=135.0,
            altitude=60.0,
            z_factor=2.0,
            scaled=True,
        )

        result = op.execute([art], params)

        lineage = result.artifact.lineage
        assert lineage.operation == "hillshade"
        assert lineage.params["azimuth"] == 135.0
        assert lineage.params["altitude"] == 60.0
        assert lineage.params["z_factor"] == 2.0
        assert lineage.params["scaled"] is True
