"""WaterElevationMosaicOperator pressure test.

Lane: operator

Stress points:
1. Protocol compliance (name, spec, declared_checks, isinstance(Operator))
2. Input validation (min_inputs=2, materialized, CRS, params type, aggregation, threshold)
3. Basic mosaic with 1 PIXC pass produces correct 3-band output (wse, confidence, mask)
4. Multiple PIXC passes with median aggregation
5. Mean and max aggregation methods
6. water_freq_threshold filters correctly (high threshold = fewer water pixels)
7. Iterative dilation fill propagates heights across water mask
8. Confidence band counts valid observations per pixel
9. All 4 declared checks pass on valid output (extent_sane, crs_valid, min_observations, backing_accessible)
10. Lineage records correct params and input IDs
11. Output artifact has correct CRS (EPSG:4326), extent, band_count=3
"""

import numpy as np
import pytest
import rasterio
from quarry_core.artifact import (
    Artifact,
    ArtifactType,
    BackingStore,
    BackingStoreKind,
    SpatialDescriptor,
    ValidationState,
    content_hash,
)
from quarry_core.operator import Operator, OperatorError, ResourceScale
from quarry_operators.water_elevation_mosaic import (
    WaterElevationMosaicOperator,
    WaterElevationMosaicParams,
)
from rasterio.crs import CRS
from rasterio.transform import from_bounds

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _write_raster(path, data, bands=1, crs=None, transform=None, nodata=None, descriptions=None):
    """Write a GeoTIFF raster.

    Args:
        path: Output file path
        data: numpy array (2D for single band, 3D for multi-band)
        bands: Number of bands (if data is 2D, this band count is used)
        crs: CRS (default EPSG:4326)
        transform: Affine transform (default from_bounds)
        nodata: Nodata value
        descriptions: List of band descriptions
    """
    if crs is None:
        crs = CRS.from_epsg(4326)

    if data.ndim == 2:
        height, width = data.shape
        # If single 2D array but bands > 1, replicate
        if bands > 1:
            data = np.stack([data] * bands, axis=0)
        else:
            data = data[np.newaxis, :, :]
    else:
        bands, height, width = data.shape

    if transform is None:
        # Default: 10x10 degree extent centered at 0,0
        transform = from_bounds(-5.0, -5.0, 5.0, 5.0, width, height)

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": bands,
        "dtype": str(data.dtype),
        "crs": crs,
        "transform": transform,
        "compress": "deflate",
    }
    if nodata is not None:
        profile["nodata"] = nodata

    with rasterio.open(path, "w", **profile) as dst:
        for i in range(bands):
            dst.write(data[i], i + 1)
        if descriptions:
            for i, desc in enumerate(descriptions):
                if desc:
                    dst.set_band_description(i + 1, desc)


def _make_artifact(path):
    """Create an Artifact for a local raster file."""
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


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def op():
    return WaterElevationMosaicOperator()


@pytest.fixture
def fof_raster(tmp_path):
    """Create FOF water frequency raster (single band, 0-1 values)."""
    # 20x20 grid with water_frequency values
    data = np.zeros((20, 20), dtype=np.float32)
    # Create water region (values > 0.3) in center
    data[5:15, 5:15] = 0.8  # High water frequency
    data[7:13, 7:13] = 0.5  # Medium water frequency
    # Land region (values < 0.3) at edges
    data[0:5, :] = 0.1
    data[15:20, :] = 0.1
    data[:, 0:5] = 0.1
    data[:, 15:20] = 0.1

    path = tmp_path / "fof_water_freq.tif"
    _write_raster(path, data, bands=1, descriptions=["water_frequency"])
    return path, data


@pytest.fixture
def pixc_raster_single(tmp_path):
    """Create single PIXC raster with 4 bands (sig0, height, water_frac, classification)."""
    # 20x20 grid
    sig0 = np.full((20, 20), 10.0, dtype=np.float32)
    height = np.full((20, 20), np.nan, dtype=np.float32)
    water_frac = np.full((20, 20), 1.0, dtype=np.float32)
    classification = np.full((20, 20), 1, dtype=np.float32)

    # Add valid heights only in water region (center)
    height[7:13, 7:13] = 100.0  # Water surface elevation
    height[8:12, 8:12] = 101.0  # Slightly higher in center

    data = np.stack([sig0, height, water_frac, classification], axis=0)

    path = tmp_path / "pixc_01.tif"
    _write_raster(
        path, data, bands=4, descriptions=["sig0", "height", "water_frac", "classification"]
    )
    return path, data


@pytest.fixture
def pixc_raster_sparse(tmp_path):
    """Create sparse PIXC raster with few valid height values."""
    sig0 = np.full((20, 20), 10.0, dtype=np.float32)
    height = np.full((20, 20), np.nan, dtype=np.float32)
    water_frac = np.full((20, 20), 1.0, dtype=np.float32)
    classification = np.full((20, 20), 1, dtype=np.float32)

    # Very sparse valid heights - only a few pixels
    height[9, 9] = 100.0
    height[9, 10] = 100.5
    height[10, 9] = 99.5
    height[10, 10] = 100.0

    data = np.stack([sig0, height, water_frac, classification], axis=0)

    path = tmp_path / "pixc_sparse.tif"
    _write_raster(
        path, data, bands=4, descriptions=["sig0", "height", "water_frac", "classification"]
    )
    return path, data


@pytest.fixture
def pixc_raster_multiple(tmp_path):
    """Create multiple PIXC rasters for multi-pass testing."""
    paths = []
    for i in range(3):
        sig0 = np.full((20, 20), 10.0 + i, dtype=np.float32)
        height = np.full((20, 20), np.nan, dtype=np.float32)
        water_frac = np.full((20, 20), 1.0, dtype=np.float32)
        classification = np.full((20, 20), 1, dtype=np.float32)

        # Different heights for each pass
        height[7:13, 7:13] = 100.0 + i * 2.0  # 100, 102, 104

        data = np.stack([sig0, height, water_frac, classification], axis=0)

        path = tmp_path / f"pixc_{i:02d}.tif"
        _write_raster(
            path, data, bands=4, descriptions=["sig0", "height", "water_frac", "classification"]
        )
        paths.append((path, data))

    return paths


@pytest.fixture
def fof_high_threshold(tmp_path):
    """FOF raster with varying water frequencies for threshold testing."""
    data = np.zeros((20, 20), dtype=np.float32)
    # Very high water frequency (0.9) in center
    data[8:12, 8:12] = 0.9
    # Medium-high (0.7) around that
    data[6:14, 6:14] = np.where(data[6:14, 6:14] == 0, 0.7, data[6:14, 6:14])
    # Medium (0.5) further out
    data[4:16, 4:16] = np.where(data[4:16, 4:16] == 0, 0.5, data[4:16, 4:16])
    # Low (0.2) at edges
    data[data == 0] = 0.2

    path = tmp_path / "fof_threshold.tif"
    _write_raster(path, data, bands=1)
    return path, data


# -----------------------------------------------------------------------------
# Protocol Compliance
# -----------------------------------------------------------------------------


class TestProtocolCompliance:
    def test_implements_operator_protocol(self, op):
        assert isinstance(op, Operator)

    def test_name(self, op):
        assert op.name == "water_elevation_mosaic"

    def test_spec(self, op):
        spec = op.spec
        assert spec.input_types == (ArtifactType.RASTER,)
        assert spec.output_type == ArtifactType.RASTER
        assert spec.min_inputs == 2
        assert spec.max_inputs == -1  # unbounded
        assert spec.resource_scale == ResourceScale.HEAVY

    def test_declared_checks(self, op):
        checks = op.declared_checks()
        assert "extent_sane" in checks
        assert "crs_valid" in checks
        assert "min_observations" in checks
        assert "backing_accessible" in checks
        assert len(checks) == 4


# -----------------------------------------------------------------------------
# Input Validation
# -----------------------------------------------------------------------------


class TestValidation:
    def test_validate_too_few_inputs(self, op, fof_raster):
        """Must have at least 2 inputs (fof + at least 1 PIXC)."""
        path, _ = fof_raster
        art = _make_artifact(path)
        errors = op.validate_inputs([art], WaterElevationMosaicParams(output_path="/tmp/x.tif"))
        assert any("at least 2" in e.lower() for e in errors)

    def test_validate_empty_inputs(self, op):
        """Empty input list should fail."""
        errors = op.validate_inputs([], WaterElevationMosaicParams(output_path="/tmp/x.tif"))
        assert any("at least 2" in e.lower() for e in errors)

    def test_validate_non_materialized_fof(self, op, fof_raster, pixc_raster_single):
        """FOF artifact must be materialized."""
        fof_path, _ = fof_raster
        pixc_path, _ = pixc_raster_single

        fof_art = _make_artifact(fof_path)
        # Create lazy FOF artifact
        lazy_fof = Artifact(
            type=fof_art.type,
            name=fof_art.name,
            backing=BackingStore(
                kind=BackingStoreKind.LAZY_HANDLE,
                uri=fof_art.backing.uri,
                size_bytes=fof_art.backing.size_bytes,
                content_hash=fof_art.backing.content_hash,
            ),
            spatial=fof_art.spatial,
        )

        pixc_art = _make_artifact(pixc_path)
        errors = op.validate_inputs(
            [lazy_fof, pixc_art], WaterElevationMosaicParams(output_path="/tmp/x.tif")
        )
        assert any("not materialized" in e.lower() for e in errors)

    def test_validate_non_materialized_pixc(self, op, fof_raster, pixc_raster_single):
        """PIXC artifact must be materialized."""
        fof_path, _ = fof_raster
        pixc_path, _ = pixc_raster_single

        fof_art = _make_artifact(fof_path)
        pixc_art = _make_artifact(pixc_path)

        # Create lazy PIXC artifact
        lazy_pixc = Artifact(
            type=pixc_art.type,
            name=pixc_art.name,
            backing=BackingStore(
                kind=BackingStoreKind.LAZY_HANDLE,
                uri=pixc_art.backing.uri,
                size_bytes=pixc_art.backing.size_bytes,
                content_hash=pixc_art.backing.content_hash,
            ),
            spatial=pixc_art.spatial,
        )

        errors = op.validate_inputs(
            [fof_art, lazy_pixc], WaterElevationMosaicParams(output_path="/tmp/x.tif")
        )
        assert any("not materialized" in e.lower() for e in errors)

    def test_validate_missing_crs_fof(self, op, fof_raster, pixc_raster_single, tmp_path):
        """FOF artifact must have CRS."""
        fof_path, _ = fof_raster
        pixc_path, _ = pixc_raster_single

        fof_art = _make_artifact(fof_path)
        pixc_art = _make_artifact(pixc_path)

        # Create FOF artifact without CRS
        no_crs_fof = Artifact(
            type=fof_art.type,
            name=fof_art.name,
            backing=fof_art.backing,
            spatial=SpatialDescriptor(
                crs=None,
                extent=fof_art.spatial.extent,
                resolution=fof_art.spatial.resolution,
            ),
        )

        errors = op.validate_inputs(
            [no_crs_fof, pixc_art], WaterElevationMosaicParams(output_path="/tmp/x.tif")
        )
        assert any("no crs" in e.lower() for e in errors)

    def test_validate_missing_crs_pixc(self, op, fof_raster, pixc_raster_single):
        """PIXC artifact must have CRS."""
        fof_path, _ = fof_raster
        pixc_path, _ = pixc_raster_single

        fof_art = _make_artifact(fof_path)
        pixc_art = _make_artifact(pixc_path)

        # Create PIXC artifact without CRS
        no_crs_pixc = Artifact(
            type=pixc_art.type,
            name=pixc_art.name,
            backing=pixc_art.backing,
            spatial=SpatialDescriptor(
                crs=None,
                extent=pixc_art.spatial.extent,
                resolution=pixc_art.spatial.resolution,
            ),
        )

        errors = op.validate_inputs(
            [fof_art, no_crs_pixc], WaterElevationMosaicParams(output_path="/tmp/x.tif")
        )
        assert any("no crs" in e.lower() for e in errors)

    def test_validate_wrong_params_type(self, op, fof_raster, pixc_raster_single):
        """Params must be WaterElevationMosaicParams."""
        fof_path, _ = fof_raster
        pixc_path, _ = pixc_raster_single

        fof_art = _make_artifact(fof_path)
        pixc_art = _make_artifact(pixc_path)

        # Use wrong params type
        class WrongParams:
            output_path = "/tmp/x.tif"

        errors = op.validate_inputs([fof_art, pixc_art], WrongParams())
        assert any("waterelevationmosaicparams" in e.lower() for e in errors)

    def test_validate_missing_output_path(self, op, fof_raster, pixc_raster_single):
        """output_path is required."""
        fof_path, _ = fof_raster
        pixc_path, _ = pixc_raster_single

        fof_art = _make_artifact(fof_path)
        pixc_art = _make_artifact(pixc_path)

        errors = op.validate_inputs([fof_art, pixc_art], WaterElevationMosaicParams(output_path=""))
        assert any("output_path" in e.lower() for e in errors)

    @pytest.mark.parametrize("aggregation", ["invalid", "sum", "min", "average"])
    def test_validate_invalid_aggregation(self, op, fof_raster, pixc_raster_single, aggregation):
        """Aggregation must be median, mean, or max."""
        fof_path, _ = fof_raster
        pixc_path, _ = pixc_raster_single

        fof_art = _make_artifact(fof_path)
        pixc_art = _make_artifact(pixc_path)

        errors = op.validate_inputs(
            [fof_art, pixc_art],
            WaterElevationMosaicParams(output_path="/tmp/x.tif", aggregation=aggregation),
        )
        assert any("aggregation" in e.lower() for e in errors)

    @pytest.mark.parametrize("threshold", [-0.1, 1.1, 2.0, -1.0])
    def test_validate_invalid_threshold_low(self, op, fof_raster, pixc_raster_single, threshold):
        """water_freq_threshold must be 0-1."""
        fof_path, _ = fof_raster
        pixc_path, _ = pixc_raster_single

        fof_art = _make_artifact(fof_path)
        pixc_art = _make_artifact(pixc_path)

        errors = op.validate_inputs(
            [fof_art, pixc_art],
            WaterElevationMosaicParams(output_path="/tmp/x.tif", water_freq_threshold=threshold),
        )
        assert any("threshold" in e.lower() for e in errors)

    @pytest.mark.parametrize("threshold", [0.0, 0.5, 1.0])
    def test_validate_valid_threshold_boundaries(
        self, op, fof_raster, pixc_raster_single, threshold
    ):
        """Threshold boundaries 0.0 and 1.0 should be valid."""
        fof_path, _ = fof_raster
        pixc_path, _ = pixc_raster_single

        fof_art = _make_artifact(fof_path)
        pixc_art = _make_artifact(pixc_path)

        errors = op.validate_inputs(
            [fof_art, pixc_art],
            WaterElevationMosaicParams(output_path="/tmp/x.tif", water_freq_threshold=threshold),
        )
        assert errors == []

    def test_accepts_valid_input(self, op, fof_raster, pixc_raster_single):
        """Valid inputs should pass validation."""
        fof_path, _ = fof_raster
        pixc_path, _ = pixc_raster_single

        fof_art = _make_artifact(fof_path)
        pixc_art = _make_artifact(pixc_path)

        errors = op.validate_inputs(
            [fof_art, pixc_art], WaterElevationMosaicParams(output_path="/tmp/x.tif")
        )
        assert errors == []


# -----------------------------------------------------------------------------
# Execute - Basic Mosaic
# -----------------------------------------------------------------------------


class TestExecuteBasic:
    def test_basic_mosaic_single_pixc(self, op, fof_raster, pixc_raster_single, tmp_path):
        """Basic mosaic with 1 PIXC produces 3-band output."""
        fof_path, fof_data = fof_raster
        pixc_path, pixc_data = pixc_raster_single

        fof_art = _make_artifact(fof_path)
        pixc_art = _make_artifact(pixc_path)

        output = tmp_path / "wse_mosaic.tif"
        params = WaterElevationMosaicParams(output_path=str(output))

        result = op.execute([fof_art, pixc_art], params)

        # Verify output file exists
        assert output.exists()

        # Verify 3 bands
        with rasterio.open(output) as src:
            assert src.count == 3
            assert src.dtypes[0] == "float32"

            wse = src.read(1)
            confidence = src.read(2)
            mask = src.read(3)

            # Check band descriptions
            assert src.descriptions[0] == "wse"
            assert src.descriptions[1] == "confidence"
            assert src.descriptions[2] == "mask"

            # Mask should be binary (0 or 1)
            assert np.all((mask == 0) | (mask == 1))

            # Water pixels should have mask=1
            water_pixels = mask == 1
            assert np.any(water_pixels)

            # Confidence should be 0 or 1 for single PIXC
            assert np.all((confidence == 0) | (confidence == 1))

    def test_output_crs_is_epsg4326(self, op, fof_raster, pixc_raster_single, tmp_path):
        """Output CRS should be EPSG:4326."""
        fof_path, _ = fof_raster
        pixc_path, _ = pixc_raster_single

        fof_art = _make_artifact(fof_path)
        pixc_art = _make_artifact(pixc_path)

        output = tmp_path / "wse_mosaic.tif"
        params = WaterElevationMosaicParams(output_path=str(output))

        result = op.execute([fof_art, pixc_art], params)

        with rasterio.open(output) as src:
            assert "4326" in str(src.crs)

        assert "4326" in str(result.artifact.spatial.crs)

    def test_output_extent_matches_fof(self, op, fof_raster, pixc_raster_single, tmp_path):
        """Output extent should match FOF input."""
        fof_path, _ = fof_raster
        pixc_path, _ = pixc_raster_single

        fof_art = _make_artifact(fof_path)
        pixc_art = _make_artifact(pixc_path)

        output = tmp_path / "wse_mosaic.tif"
        params = WaterElevationMosaicParams(output_path=str(output))

        result = op.execute([fof_art, pixc_art], params)

        with rasterio.open(output) as src:
            output_bounds = src.bounds

        fof_bounds = rasterio.open(fof_path).bounds

        assert abs(output_bounds.left - fof_bounds.left) < 1e-6
        assert abs(output_bounds.right - fof_bounds.right) < 1e-6
        assert abs(output_bounds.bottom - fof_bounds.bottom) < 1e-6
        assert abs(output_bounds.top - fof_bounds.top) < 1e-6

    def test_output_band_count_in_artifact(self, op, fof_raster, pixc_raster_single, tmp_path):
        """Output artifact should have band_count=3."""
        fof_path, _ = fof_raster
        pixc_path, _ = pixc_raster_single

        fof_art = _make_artifact(fof_path)
        pixc_art = _make_artifact(pixc_path)

        output = tmp_path / "wse_mosaic.tif"
        params = WaterElevationMosaicParams(output_path=str(output))

        result = op.execute([fof_art, pixc_art], params)

        assert result.artifact.spatial.band_count == 3


# -----------------------------------------------------------------------------
# Execute - Multiple PIXC Passes
# -----------------------------------------------------------------------------


class TestExecuteMultiplePixc:
    def test_multiple_pixc_median_aggregation(self, op, fof_raster, pixc_raster_multiple, tmp_path):
        """Multiple PIXC passes with median aggregation."""
        fof_path, _ = fof_raster

        fof_art = _make_artifact(fof_path)
        pixc_arts = [_make_artifact(p) for p, _ in pixc_raster_multiple]

        output = tmp_path / "wse_mosaic.tif"
        params = WaterElevationMosaicParams(output_path=str(output), aggregation="median")

        result = op.execute([fof_art] + pixc_arts, params)

        with rasterio.open(output) as src:
            wse = src.read(1)
            confidence = src.read(2)
            mask = src.read(3)

            # Confidence should be up to 3 (3 PIXC inputs)
            assert np.max(confidence) == 3

            # Water pixels should have valid WSE
            water_mask = mask == 1
            assert np.all(np.isfinite(wse[water_mask]))

            # Median of [100, 102, 104] = 102
            # Check that center pixels have median value
            center_wse = wse[8:12, 8:12]
            assert np.all(center_wse[center_wse > 0] == 102.0)

    def test_mean_aggregation(self, op, fof_raster, pixc_raster_multiple, tmp_path):
        """Mean aggregation across multiple PIXC passes."""
        fof_path, _ = fof_raster

        fof_art = _make_artifact(fof_path)
        pixc_arts = [_make_artifact(p) for p, _ in pixc_raster_multiple]

        output = tmp_path / "wse_mosaic.tif"
        params = WaterElevationMosaicParams(output_path=str(output), aggregation="mean")

        result = op.execute([fof_art] + pixc_arts, params)

        with rasterio.open(output) as src:
            wse = src.read(1)

            # Mean of [100, 102, 104] = 102
            center_wse = wse[8:12, 8:12]
            assert np.allclose(center_wse[center_wse > 0], 102.0, rtol=0.01)

    def test_max_aggregation(self, op, fof_raster, pixc_raster_multiple, tmp_path):
        """Max aggregation across multiple PIXC passes."""
        fof_path, _ = fof_raster

        fof_art = _make_artifact(fof_path)
        pixc_arts = [_make_artifact(p) for p, _ in pixc_raster_multiple]

        output = tmp_path / "wse_mosaic.tif"
        params = WaterElevationMosaicParams(output_path=str(output), aggregation="max")

        result = op.execute([fof_art] + pixc_arts, params)

        with rasterio.open(output) as src:
            wse = src.read(1)

            # Max of [100, 102, 104] = 104
            center_wse = wse[8:12, 8:12]
            assert np.all(center_wse[center_wse > 0] == 104.0)


# -----------------------------------------------------------------------------
# Execute - Water Frequency Threshold
# -----------------------------------------------------------------------------


class TestExecuteThreshold:
    def test_high_threshold_fewer_water_pixels(
        self, op, fof_high_threshold, pixc_raster_single, tmp_path
    ):
        """Higher threshold should result in fewer water pixels."""
        fof_path, _ = fof_high_threshold
        pixc_path, _ = pixc_raster_single

        fof_art = _make_artifact(fof_path)
        pixc_art = _make_artifact(pixc_path)

        # Low threshold (0.3)
        output_low = tmp_path / "wse_low_thresh.tif"
        params_low = WaterElevationMosaicParams(
            output_path=str(output_low), water_freq_threshold=0.3
        )
        op.execute([fof_art, pixc_art], params_low)

        # High threshold (0.8)
        output_high = tmp_path / "wse_high_thresh.tif"
        params_high = WaterElevationMosaicParams(
            output_path=str(output_high), water_freq_threshold=0.8
        )
        op.execute([fof_art, pixc_art], params_high)

        with rasterio.open(output_low) as src_low, rasterio.open(output_high) as src_high:
            mask_low = src_low.read(3)
            mask_high = src_high.read(3)

            water_pixels_low = np.sum(mask_low == 1)
            water_pixels_high = np.sum(mask_high == 1)

            # High threshold should have fewer water pixels
            assert water_pixels_high < water_pixels_low

    def test_threshold_zero_all_water(self, op, fof_raster, pixc_raster_single, tmp_path):
        """Threshold of 0 should classify all pixels as water."""
        fof_path, fof_data = fof_raster
        pixc_path, _ = pixc_raster_single

        fof_art = _make_artifact(fof_path)
        pixc_art = _make_artifact(pixc_path)

        output = tmp_path / "wse_mosaic.tif"
        params = WaterElevationMosaicParams(output_path=str(output), water_freq_threshold=0.0)

        op.execute([fof_art, pixc_art], params)

        with rasterio.open(output) as src:
            mask = src.read(3)
            # All pixels should be water (mask=1)
            assert np.all(mask == 1)

    def test_threshold_one_only_certain_water(
        self, op, fof_high_threshold, pixc_raster_single, tmp_path
    ):
        """Threshold of 1.0 should only include pixels with water_freq=1.0."""
        fof_path, fof_data = fof_high_threshold
        pixc_path, _ = pixc_raster_single

        fof_art = _make_artifact(fof_path)
        pixc_art = _make_artifact(pixc_path)

        output = tmp_path / "wse_mosaic.tif"
        params = WaterElevationMosaicParams(output_path=str(output), water_freq_threshold=1.0)

        op.execute([fof_art, pixc_art], params)

        with rasterio.open(output) as src:
            mask = src.read(3)
            # Only pixels with water_freq >= 1.0 should be water
            # In our fixture, no pixels have exactly 1.0
            assert np.all(mask == 0)


# -----------------------------------------------------------------------------
# Execute - Iterative Dilation Fill
# -----------------------------------------------------------------------------


class TestExecuteDilationFill:
    def test_dilation_fill_propagates_heights(self, op, fof_raster, pixc_raster_sparse, tmp_path):
        """Iterative dilation should propagate sparse heights across water mask."""
        fof_path, fof_data = fof_raster
        pixc_path, pixc_data = pixc_raster_sparse

        fof_art = _make_artifact(fof_path)
        pixc_art = _make_artifact(pixc_path)

        output = tmp_path / "wse_mosaic.tif"
        params = WaterElevationMosaicParams(output_path=str(output))

        op.execute([fof_art, pixc_art], params)

        with rasterio.open(output) as src:
            wse = src.read(1)
            mask = src.read(3)

            # All water pixels should have valid WSE after fill
            water_mask = mask == 1
            water_wse = wse[water_mask]

            # All water pixels should have finite values
            assert np.all(np.isfinite(water_wse))

            # Original sparse pixels had values around 100
            # Filled pixels should have interpolated values
            assert np.all(water_wse > 0)

    def test_dilation_fill_with_no_valid_heights(self, op, fof_raster, tmp_path):
        """Dilation with no valid heights should produce NaN in water mask."""
        fof_path, fof_data = fof_raster

        fof_art = _make_artifact(fof_path)

        # Create PIXC with all NaN heights
        sig0 = np.full((20, 20), 10.0, dtype=np.float32)
        height = np.full((20, 20), np.nan, dtype=np.float32)
        water_frac = np.full((20, 20), 1.0, dtype=np.float32)
        classification = np.full((20, 20), 1, dtype=np.float32)

        data = np.stack([sig0, height, water_frac, classification], axis=0)

        pixc_path = tmp_path / "pixc_all_nan.tif"
        _write_raster(pixc_path, data, bands=4)
        pixc_art = _make_artifact(pixc_path)

        output = tmp_path / "wse_mosaic.tif"
        params = WaterElevationMosaicParams(output_path=str(output))

        op.execute([fof_art, pixc_art], params)

        with rasterio.open(output) as src:
            wse = src.read(1)
            mask = src.read(3)

            # Water pixels should still exist in mask
            water_mask = mask == 1
            assert np.any(water_mask)

            # But WSE should be NaN (no valid heights to fill from)
            water_wse = wse[water_mask]
            assert np.all(np.isnan(water_wse))


# -----------------------------------------------------------------------------
# Execute - Confidence Band
# -----------------------------------------------------------------------------


class TestExecuteConfidence:
    def test_confidence_counts_valid_observations(
        self, op, fof_raster, pixc_raster_multiple, tmp_path
    ):
        """Confidence band should count valid observations per pixel."""
        fof_path, _ = fof_raster

        fof_art = _make_artifact(fof_path)
        pixc_arts = [_make_artifact(p) for p, _ in pixc_raster_multiple]

        output = tmp_path / "wse_mosaic.tif"
        params = WaterElevationMosaicParams(output_path=str(output))

        op.execute([fof_art] + pixc_arts, params)

        with rasterio.open(output) as src:
            confidence = src.read(2)
            mask = src.read(3)

            # Water pixels in center should have confidence=3 (all 3 PIXC have values)
            assert np.all(confidence[8:12, 8:12] == 3)

            # Land pixels should have confidence=0
            land_mask = mask == 0
            assert np.all(confidence[land_mask] == 0)

    def test_confidence_partial_coverage(self, op, fof_raster, tmp_path):
        """Confidence should reflect partial PIXC coverage."""
        fof_path, _ = fof_raster
        fof_art = _make_artifact(fof_path)

        # Create PIXC with different coverage areas
        pixc_arts = []
        for i in range(3):
            sig0 = np.full((20, 20), 10.0, dtype=np.float32)
            height = np.full((20, 20), np.nan, dtype=np.float32)
            water_frac = np.full((20, 20), 1.0, dtype=np.float32)
            classification = np.full((20, 20), 1, dtype=np.float32)

            # Each PIXC covers a different row
            height[7 + i, 7:13] = 100.0 + i

            data = np.stack([sig0, height, water_frac, classification], axis=0)

            pixc_path = tmp_path / f"pixc_partial_{i}.tif"
            _write_raster(pixc_path, data, bands=4)
            pixc_arts.append(_make_artifact(pixc_path))

        output = tmp_path / "wse_mosaic.tif"
        params = WaterElevationMosaicParams(output_path=str(output))

        op.execute([fof_art] + pixc_arts, params)

        with rasterio.open(output) as src:
            confidence = src.read(2)

            # Row 7 should have confidence=1 (only first PIXC)
            assert np.all(confidence[7, 7:13] == 1)

            # Row 8 should have confidence=1 (only second PIXC)
            assert np.all(confidence[8, 7:13] == 1)

            # Row 9 should have confidence=1 (only third PIXC)
            assert np.all(confidence[9, 7:13] == 1)


# -----------------------------------------------------------------------------
# Checks
# -----------------------------------------------------------------------------


class TestChecks:
    def test_all_checks_pass_happy_path(self, op, fof_raster, pixc_raster_single, tmp_path):
        """All 4 checks should pass on valid output."""
        fof_path, _ = fof_raster
        pixc_path, _ = pixc_raster_single

        fof_art = _make_artifact(fof_path)
        pixc_art = _make_artifact(pixc_path)

        output = tmp_path / "wse_mosaic.tif"
        params = WaterElevationMosaicParams(output_path=str(output))

        result = op.execute([fof_art, pixc_art], params)

        check_names = {c.check_name: c for c in result.checks}

        # All 4 declared checks should be present
        assert "extent_sane" in check_names
        assert "crs_valid" in check_names
        assert "min_observations" in check_names
        assert "backing_accessible" in check_names

        # All should be VALID
        for check in result.checks:
            assert check.state == ValidationState.VALID, f"{check.check_name}: {check.message}"

    def test_extent_sane_check(self, op, fof_raster, pixc_raster_single, tmp_path):
        """extent_sane check should validate output extent."""
        fof_path, _ = fof_raster
        pixc_path, _ = pixc_raster_single

        fof_art = _make_artifact(fof_path)
        pixc_art = _make_artifact(pixc_path)

        output = tmp_path / "wse_mosaic.tif"
        params = WaterElevationMosaicParams(output_path=str(output))

        result = op.execute([fof_art, pixc_art], params)

        extent_check = next(c for c in result.checks if c.check_name == "extent_sane")
        assert extent_check.state == ValidationState.VALID
        assert "extent" in extent_check.message.lower()

    def test_crs_valid_check(self, op, fof_raster, pixc_raster_single, tmp_path):
        """crs_valid check should validate output CRS."""
        fof_path, _ = fof_raster
        pixc_path, _ = pixc_raster_single

        fof_art = _make_artifact(fof_path)
        pixc_art = _make_artifact(pixc_path)

        output = tmp_path / "wse_mosaic.tif"
        params = WaterElevationMosaicParams(output_path=str(output))

        result = op.execute([fof_art, pixc_art], params)

        crs_check = next(c for c in result.checks if c.check_name == "crs_valid")
        assert crs_check.state == ValidationState.VALID
        assert "4326" in crs_check.message

    def test_min_observations_check(self, op, fof_raster, pixc_raster_single, tmp_path):
        """min_observations check should report coverage."""
        fof_path, _ = fof_raster
        pixc_path, _ = pixc_raster_single

        fof_art = _make_artifact(fof_path)
        pixc_art = _make_artifact(pixc_path)

        output = tmp_path / "wse_mosaic.tif"
        params = WaterElevationMosaicParams(output_path=str(output))

        result = op.execute([fof_art, pixc_art], params)

        obs_check = next(c for c in result.checks if c.check_name == "min_observations")
        assert obs_check.state == ValidationState.VALID
        assert "coverage" in obs_check.message.lower()

    def test_backing_accessible_check(self, op, fof_raster, pixc_raster_single, tmp_path):
        """backing_accessible check should verify file exists."""
        fof_path, _ = fof_raster
        pixc_path, _ = pixc_raster_single

        fof_art = _make_artifact(fof_path)
        pixc_art = _make_artifact(pixc_path)

        output = tmp_path / "wse_mosaic.tif"
        params = WaterElevationMosaicParams(output_path=str(output))

        result = op.execute([fof_art, pixc_art], params)

        backing_check = next(c for c in result.checks if c.check_name == "backing_accessible")
        assert backing_check.state == ValidationState.VALID
        assert "file exists" in backing_check.message.lower()


# -----------------------------------------------------------------------------
# Lineage and Metadata
# -----------------------------------------------------------------------------


class TestLineageAndMetadata:
    def test_lineage_records_operation(self, op, fof_raster, pixc_raster_single, tmp_path):
        """Lineage should record operation name."""
        fof_path, _ = fof_raster
        pixc_path, _ = pixc_raster_single

        fof_art = _make_artifact(fof_path)
        pixc_art = _make_artifact(pixc_path)

        output = tmp_path / "wse_mosaic.tif"
        params = WaterElevationMosaicParams(output_path=str(output))

        result = op.execute([fof_art, pixc_art], params)

        assert result.artifact.lineage.operation == "water_elevation_mosaic"

    def test_lineage_records_input_ids(self, op, fof_raster, pixc_raster_single, tmp_path):
        """Lineage should record all input artifact IDs."""
        fof_path, _ = fof_raster
        pixc_path, _ = pixc_raster_single

        fof_art = _make_artifact(fof_path)
        pixc_art = _make_artifact(pixc_path)

        output = tmp_path / "wse_mosaic.tif"
        params = WaterElevationMosaicParams(output_path=str(output))

        result = op.execute([fof_art, pixc_art], params)

        input_ids = result.artifact.lineage.inputs
        assert len(input_ids) == 2
        assert fof_art.id in input_ids
        assert pixc_art.id in input_ids

    def test_lineage_records_params(self, op, fof_raster, pixc_raster_single, tmp_path):
        """Lineage should record operation parameters."""
        fof_path, _ = fof_raster
        pixc_path, _ = pixc_raster_single

        fof_art = _make_artifact(fof_path)
        pixc_art = _make_artifact(pixc_path)

        output = tmp_path / "wse_mosaic.tif"
        params = WaterElevationMosaicParams(
            output_path=str(output), water_freq_threshold=0.5, aggregation="mean"
        )

        result = op.execute([fof_art, pixc_art], params)

        lineage_params = result.artifact.lineage.params
        assert lineage_params["water_freq_threshold"] == 0.5
        assert lineage_params["aggregation"] == "mean"
        assert lineage_params["num_pixc_inputs"] == 1

    def test_metadata_contains_bands(self, op, fof_raster, pixc_raster_single, tmp_path):
        """Metadata should contain band names."""
        fof_path, _ = fof_raster
        pixc_path, _ = pixc_raster_single

        fof_art = _make_artifact(fof_path)
        pixc_art = _make_artifact(pixc_path)

        output = tmp_path / "wse_mosaic.tif"
        params = WaterElevationMosaicParams(output_path=str(output))

        result = op.execute([fof_art, pixc_art], params)

        assert "bands" in result.artifact.metadata
        assert list(result.artifact.metadata["bands"]) == ["wse", "confidence", "mask"]

    def test_metadata_contains_params(self, op, fof_raster, pixc_raster_single, tmp_path):
        """Metadata should contain operation parameters."""
        fof_path, _ = fof_raster
        pixc_path, _ = pixc_raster_single

        fof_art = _make_artifact(fof_path)
        pixc_art = _make_artifact(pixc_path)

        output = tmp_path / "wse_mosaic.tif"
        params = WaterElevationMosaicParams(
            output_path=str(output), water_freq_threshold=0.4, aggregation="max"
        )

        result = op.execute([fof_art, pixc_art], params)

        assert result.artifact.metadata["aggregation"] == "max"
        assert result.artifact.metadata["water_freq_threshold"] == 0.4


# -----------------------------------------------------------------------------
# Edge Cases
# -----------------------------------------------------------------------------


class TestEdgeCases:
    def test_fof_band_index_parameter(self, op, fof_raster, pixc_raster_single, tmp_path):
        """fof_band_index parameter should select correct band."""
        fof_path, _ = fof_raster
        pixc_path, _ = pixc_raster_single

        fof_art = _make_artifact(fof_path)
        pixc_art = _make_artifact(pixc_path)

        output = tmp_path / "wse_mosaic.tif"
        params = WaterElevationMosaicParams(
            output_path=str(output),
            fof_band_index=1,  # Default
        )

        result = op.execute([fof_art, pixc_art], params)
        assert result.artifact is not None

    def test_pixc_with_single_band_fails(self, op, fof_raster, tmp_path):
        """PIXC raster with only 1 band should fail."""
        fof_path, _ = fof_raster
        fof_art = _make_artifact(fof_path)

        # Create single-band PIXC (invalid)
        height = np.full((20, 20), 100.0, dtype=np.float32)
        pixc_path = tmp_path / "pixc_single_band.tif"
        _write_raster(pixc_path, height, bands=1)
        pixc_art = _make_artifact(pixc_path)

        output = tmp_path / "wse_mosaic.tif"
        params = WaterElevationMosaicParams(output_path=str(output))

        with pytest.raises(OperatorError) as exc_info:
            op.execute([fof_art, pixc_art], params)

        assert "band" in str(exc_info.value).lower()

    def test_different_grid_sizes_resampled(self, op, fof_raster, pixc_raster_single, tmp_path):
        """Different grid sizes should be resampled."""
        fof_path, _ = fof_raster
        fof_art = _make_artifact(fof_path)

        # Create smaller PIXC raster
        sig0 = np.full((10, 10), 10.0, dtype=np.float32)
        height = np.full((10, 10), 100.0, dtype=np.float32)
        water_frac = np.full((10, 10), 1.0, dtype=np.float32)
        classification = np.full((10, 10), 1, dtype=np.float32)

        data = np.stack([sig0, height, water_frac, classification], axis=0)

        pixc_path = tmp_path / "pixc_small.tif"
        _write_raster(pixc_path, data, bands=4)
        pixc_art = _make_artifact(pixc_path)

        output = tmp_path / "wse_mosaic.tif"
        params = WaterElevationMosaicParams(output_path=str(output))

        result = op.execute([fof_art, pixc_art], params)

        # Output should match FOF dimensions
        with rasterio.open(output) as src:
            assert src.height == 20
            assert src.width == 20

    def test_many_pixc_inputs(self, op, fof_raster, tmp_path):
        """Test with many PIXC inputs (10+)."""
        fof_path, _ = fof_raster
        fof_art = _make_artifact(fof_path)

        pixc_arts = []
        for i in range(10):
            sig0 = np.full((20, 20), 10.0, dtype=np.float32)
            height = np.full((20, 20), np.nan, dtype=np.float32)
            water_frac = np.full((20, 20), 1.0, dtype=np.float32)
            classification = np.full((20, 20), 1, dtype=np.float32)

            height[7:13, 7:13] = 100.0 + i

            data = np.stack([sig0, height, water_frac, classification], axis=0)

            pixc_path = tmp_path / f"pixc_many_{i}.tif"
            _write_raster(pixc_path, data, bands=4)
            pixc_arts.append(_make_artifact(pixc_path))

        output = tmp_path / "wse_mosaic.tif"
        params = WaterElevationMosaicParams(output_path=str(output))

        result = op.execute([fof_art] + pixc_arts, params)

        with rasterio.open(output) as src:
            confidence = src.read(2)
            # Center should have confidence=10
            assert np.all(confidence[8:12, 8:12] == 10)

    def test_output_path_created(self, op, fof_raster, pixc_raster_single, tmp_path):
        """Output directory should be created if it doesn't exist."""
        fof_path, _ = fof_raster
        pixc_path, _ = pixc_raster_single

        fof_art = _make_artifact(fof_path)
        pixc_art = _make_artifact(pixc_path)

        # Use nested directory that doesn't exist
        output = tmp_path / "nested" / "deep" / "wse_mosaic.tif"
        params = WaterElevationMosaicParams(output_path=str(output))

        result = op.execute([fof_art, pixc_art], params)

        assert output.exists()
