"""BuildCOGOperator pressure test.

Lane: operator

Stress points:
1. Protocol compliance (spec, validate_inputs, declared_checks)
2. Happy path — non-COG GeoTIFF → valid COG
3. is_cog check — output has tiling + overviews
4. CRS preserved through COG build
5. Data unchanged — base resolution pixels identical to input
6. Nodata preservation (numeric and NaN)
7. Multi-band preservation
8. Already-a-COG idempotence — output still valid, data unchanged
9. Tiny raster with oversized blocksize — still produces valid output
10. Overview levels computed correctly
11. Compression applied — output smaller than uncompressed
12. Lazy artifact rejected at validation
13. Lineage records build params
14. Output artifact metadata fresh from file
15. All checks pass on happy path
16. Unsupported compression rejected
"""

from __future__ import annotations

from pathlib import Path

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
from quarry_core.operator import Operator, ResourceScale
from quarry_operators.build_cog import (
    BuildCOGOperator,
    BuildCOGParams,
)
from rasterio.crs import CRS
from rasterio.transform import from_bounds

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_raster(path, data, crs_epsg=32610, nodata=None, extent=None):
    """Write a plain (non-COG) GeoTIFF."""
    if data.ndim == 2:
        data = data[np.newaxis, :, :]
    bands, nrows, ncols = data.shape
    if extent is None:
        extent = (0, 0, ncols, nrows)
    xmin, ymin, xmax, ymax = extent
    transform = from_bounds(xmin, ymin, xmax, ymax, ncols, nrows)
    meta = {
        "driver": "GTiff",
        "height": nrows,
        "width": ncols,
        "count": bands,
        "dtype": str(data.dtype),
        "crs": CRS.from_epsg(crs_epsg),
        "transform": transform,
    }
    if nodata is not None:
        meta["nodata"] = nodata
    with rasterio.open(path, "w", **meta) as dst:
        for b in range(bands):
            dst.write(data[b], b + 1)


def _write_cog(path, data, crs_epsg=32610, nodata=None, blocksize=256):
    """Write a valid COG (tiled + overviews)."""
    if data.ndim == 2:
        data = data[np.newaxis, :, :]
    bands, nrows, ncols = data.shape
    transform = from_bounds(0, 0, ncols, nrows, ncols, nrows)
    meta = {
        "driver": "GTiff",
        "height": nrows,
        "width": ncols,
        "count": bands,
        "dtype": str(data.dtype),
        "crs": CRS.from_epsg(crs_epsg),
        "transform": transform,
        "tiled": True,
        "blockxsize": blocksize,
        "blockysize": blocksize,
        "compress": "deflate",
    }
    if nodata is not None:
        meta["nodata"] = nodata
    with rasterio.open(path, "w", **meta) as dst:
        for b in range(bands):
            dst.write(data[b], b + 1)
    # Add overviews
    with rasterio.open(path, "r+") as dst:
        dst.build_overviews([2], rasterio.enums.Resampling.nearest)
        dst.update_tags(ns="rio_overview", resampling="nearest")
    # Copy to COG layout
    from rasterio.shutil import copy as rio_copy

    cog_tmp = path.with_suffix(".cog.tif")
    rio_copy(
        path,
        cog_tmp,
        driver="GTiff",
        copy_src_overviews=True,
        tiled=True,
        blockxsize=blocksize,
        blockysize=blocksize,
        compress="deflate",
    )
    cog_tmp.replace(path)


def _make_raster_artifact(path, crs_epsg=32610):
    """Create Artifact for a raster file."""
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


def _is_cog(path):
    """Check if a file is a valid COG (tiled + overviews)."""
    with rasterio.open(path) as src:
        is_tiled = src.profile.get("tiled", False)
        overviews = src.overviews(1)
        return is_tiled and len(overviews) > 0


def _read_all_bands(path):
    """Read all bands as numpy array."""
    with rasterio.open(path) as src:
        return src.read()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def op():
    return BuildCOGOperator()


@pytest.fixture
def workspace(tmp_path):
    return tmp_path


# ---------------------------------------------------------------------------
# 1. Protocol compliance
# ---------------------------------------------------------------------------


def test_protocol_compliance(op):
    """Operator satisfies the Operator protocol."""
    assert isinstance(op, Operator)


def test_spec(op):
    spec = op.spec
    assert spec.input_types == (ArtifactType.RASTER,)
    assert spec.output_type == ArtifactType.RASTER
    assert spec.min_inputs == 1
    assert spec.max_inputs == 1
    assert spec.resource_scale == ResourceScale.MEDIUM


def test_declared_checks(op):
    checks = op.declared_checks()
    assert "is_cog" in checks
    assert "crs_preserved" in checks
    assert "dimensions_preserved" in checks
    assert "nodata_preserved" in checks


# ---------------------------------------------------------------------------
# 2. Happy path — non-COG → valid COG
# ---------------------------------------------------------------------------


def test_happy_path_non_cog_to_cog(op, workspace):
    """Plain GeoTIFF becomes a valid COG."""
    data = np.random.rand(512, 512).astype(np.float32)
    raster_path = workspace / "input.tif"
    _write_raster(raster_path, data)

    art = _make_raster_artifact(raster_path)
    assert not _is_cog(raster_path)

    params = BuildCOGParams(output_path=str(workspace / "output.tif"))
    result = op.execute([art], params)

    assert result.artifact.type == ArtifactType.RASTER
    assert _is_cog(workspace / "output.tif")
    assert result.artifact.metadata["is_cog"] is True


# ---------------------------------------------------------------------------
# 3. is_cog check passes
# ---------------------------------------------------------------------------


def test_is_cog_check_passes(op, workspace):
    """is_cog check reports VALID on output."""
    data = np.ones((512, 512), dtype=np.float32)
    raster_path = workspace / "input.tif"
    _write_raster(raster_path, data)
    art = _make_raster_artifact(raster_path)

    params = BuildCOGParams(output_path=str(workspace / "output.tif"))
    result = op.execute([art], params)

    cog_check = [c for c in result.checks if c.check_name == "is_cog"]
    assert len(cog_check) == 1
    assert cog_check[0].state == ValidationState.VALID


# ---------------------------------------------------------------------------
# 4. CRS preserved
# ---------------------------------------------------------------------------


def test_crs_preserved(op, workspace):
    """CRS survives COG build unchanged."""
    data = np.ones((512, 512), dtype=np.float32)
    raster_path = workspace / "input.tif"
    _write_raster(raster_path, data, crs_epsg=4326)
    art = _make_raster_artifact(raster_path, crs_epsg=4326)

    params = BuildCOGParams(output_path=str(workspace / "output.tif"))
    result = op.execute([art], params)

    with rasterio.open(workspace / "output.tif") as src:
        assert src.crs.to_epsg() == 4326

    crs_check = [c for c in result.checks if c.check_name == "crs_preserved"]
    assert crs_check[0].state == ValidationState.VALID


# ---------------------------------------------------------------------------
# 5. Data unchanged — pixel equality at base resolution
# ---------------------------------------------------------------------------


def test_data_unchanged(op, workspace):
    """Base resolution pixel values are identical after COG build."""
    data = np.arange(1, 512 * 512 + 1, dtype=np.float32).reshape(512, 512)
    raster_path = workspace / "input.tif"
    _write_raster(raster_path, data)
    art = _make_raster_artifact(raster_path)

    params = BuildCOGParams(output_path=str(workspace / "output.tif"))
    op.execute([art], params)

    input_data = _read_all_bands(raster_path)
    output_data = _read_all_bands(workspace / "output.tif")
    np.testing.assert_array_equal(input_data, output_data)


# ---------------------------------------------------------------------------
# 6. Nodata preservation — numeric
# ---------------------------------------------------------------------------


def test_nodata_numeric_preserved(op, workspace):
    """Numeric nodata value survives COG build."""
    data = np.array([[1, 2], [3, -9999]], dtype=np.float32)
    # Make it big enough for overviews
    big = np.tile(data, (256, 256))
    raster_path = workspace / "input.tif"
    _write_raster(raster_path, big, nodata=-9999)
    art = _make_raster_artifact(raster_path)

    params = BuildCOGParams(output_path=str(workspace / "output.tif"))
    result = op.execute([art], params)

    with rasterio.open(workspace / "output.tif") as src:
        assert src.nodata == -9999

    nodata_check = [c for c in result.checks if c.check_name == "nodata_preserved"]
    assert nodata_check[0].state == ValidationState.VALID


# ---------------------------------------------------------------------------
# 7. Nodata preservation — NaN
# ---------------------------------------------------------------------------


def test_nodata_nan_preserved(op, workspace):
    """NaN nodata value survives COG build."""
    data = np.ones((512, 512), dtype=np.float32)
    data[0, 0] = np.nan
    raster_path = workspace / "input.tif"
    _write_raster(raster_path, data, nodata=float("nan"))
    art = _make_raster_artifact(raster_path)

    params = BuildCOGParams(output_path=str(workspace / "output.tif"))
    result = op.execute([art], params)

    with rasterio.open(workspace / "output.tif") as src:
        assert np.isnan(src.nodata)

    nodata_check = [c for c in result.checks if c.check_name == "nodata_preserved"]
    assert nodata_check[0].state == ValidationState.VALID


# ---------------------------------------------------------------------------
# 8. Multi-band preservation
# ---------------------------------------------------------------------------


def test_multi_band_preserved(op, workspace):
    """Band count and per-band data survive COG build."""
    band1 = np.ones((512, 512), dtype=np.float32) * 10
    band2 = np.ones((512, 512), dtype=np.float32) * 20
    band3 = np.ones((512, 512), dtype=np.float32) * 30
    data = np.stack([band1, band2, band3])
    raster_path = workspace / "input.tif"
    _write_raster(raster_path, data)
    art = _make_raster_artifact(raster_path)

    params = BuildCOGParams(output_path=str(workspace / "output.tif"))
    result = op.execute([art], params)

    with rasterio.open(workspace / "output.tif") as src:
        assert src.count == 3

    dim_check = [c for c in result.checks if c.check_name == "dimensions_preserved"]
    assert dim_check[0].state == ValidationState.VALID

    output_data = _read_all_bands(workspace / "output.tif")
    np.testing.assert_array_equal(output_data[0], band1)
    np.testing.assert_array_equal(output_data[1], band2)
    np.testing.assert_array_equal(output_data[2], band3)


# ---------------------------------------------------------------------------
# 9. Already-a-COG idempotence
# ---------------------------------------------------------------------------


def test_already_cog_idempotent(op, workspace):
    """A valid COG passed through BuildCOG produces another valid COG with identical data."""
    data = np.random.rand(512, 512).astype(np.float32)
    cog_path = workspace / "already_cog.tif"
    _write_cog(cog_path, data)
    assert _is_cog(cog_path)

    art = _make_raster_artifact(cog_path)
    params = BuildCOGParams(output_path=str(workspace / "output.tif"))
    result = op.execute([art], params)

    assert _is_cog(workspace / "output.tif")

    input_data = _read_all_bands(cog_path)
    output_data = _read_all_bands(workspace / "output.tif")
    np.testing.assert_array_equal(input_data, output_data)

    # All checks pass
    for check in result.checks:
        assert check.state == ValidationState.VALID, f"Check {check.check_name}: {check.message}"


# ---------------------------------------------------------------------------
# 10. Tiny raster — smaller than blocksize
# ---------------------------------------------------------------------------


def test_tiny_raster_still_valid(op, workspace):
    """A 4x4 raster (smaller than any blocksize) still produces valid output."""
    data = np.arange(16, dtype=np.float32).reshape(4, 4)
    raster_path = workspace / "tiny.tif"
    _write_raster(raster_path, data)
    art = _make_raster_artifact(raster_path)

    params = BuildCOGParams(
        output_path=str(workspace / "output.tif"),
        blocksize=256,
    )
    result = op.execute([art], params)

    # Output should be tiled (even if single tile) but may lack overviews
    # since the raster is smaller than the blocksize.
    # This is acceptable — the operator should not crash.
    assert result.artifact.type == ArtifactType.RASTER
    assert Path(result.artifact.backing.uri).exists()

    # Data unchanged
    output_data = _read_all_bands(workspace / "output.tif")
    np.testing.assert_array_equal(output_data[0], data)


# ---------------------------------------------------------------------------
# 11. Overview levels computed correctly
# ---------------------------------------------------------------------------


def test_overview_levels_present(op, workspace):
    """Large raster gets overview levels."""
    data = np.ones((1024, 1024), dtype=np.float32)
    raster_path = workspace / "big.tif"
    _write_raster(raster_path, data)
    art = _make_raster_artifact(raster_path)

    params = BuildCOGParams(
        output_path=str(workspace / "output.tif"),
        blocksize=256,
    )
    op.execute([art], params)

    with rasterio.open(workspace / "output.tif") as src:
        overviews = src.overviews(1)
        assert len(overviews) >= 1
        assert 2 in overviews


# ---------------------------------------------------------------------------
# 12. Compression applied
# ---------------------------------------------------------------------------


def test_compression_applied(op, workspace):
    """Compressed COG is smaller than uncompressed equivalent."""
    data = np.zeros((512, 512), dtype=np.float32)
    raster_path = workspace / "input.tif"
    _write_raster(raster_path, data)
    art = _make_raster_artifact(raster_path)

    # Build compressed
    params_compressed = BuildCOGParams(
        output_path=str(workspace / "compressed.tif"),
        compress="deflate",
    )
    op.execute([art], params_compressed)

    # Build uncompressed
    params_none = BuildCOGParams(
        output_path=str(workspace / "uncompressed.tif"),
        compress="none",
    )
    op.execute([art], params_none)

    compressed_size = (workspace / "compressed.tif").stat().st_size
    uncompressed_size = (workspace / "uncompressed.tif").stat().st_size
    assert compressed_size < uncompressed_size


# ---------------------------------------------------------------------------
# 13. Lazy artifact rejected
# ---------------------------------------------------------------------------


def test_lazy_artifact_rejected(op, workspace):
    """Lazy (unmaterialized) artifact rejected at validation."""
    lazy_art = Artifact(
        type=ArtifactType.RASTER,
        backing=BackingStore(kind=BackingStoreKind.LAZY_HANDLE, uri="http://example.com/r.tif"),
        spatial=SpatialDescriptor(crs="EPSG:32610"),
    )
    params = BuildCOGParams(output_path=str(workspace / "output.tif"))
    errors = op.validate_inputs([lazy_art], params)
    assert any("not materialized" in e for e in errors)


# ---------------------------------------------------------------------------
# 14. Lineage records build params
# ---------------------------------------------------------------------------


def test_lineage_records_params(op, workspace):
    """Output artifact lineage includes build configuration."""
    data = np.ones((512, 512), dtype=np.float32)
    raster_path = workspace / "input.tif"
    _write_raster(raster_path, data)
    art = _make_raster_artifact(raster_path)

    params = BuildCOGParams(
        output_path=str(workspace / "output.tif"),
        blocksize=512,
        compress="lzw",
        overview_resampling="bilinear",
    )
    result = op.execute([art], params)

    lineage = result.artifact.lineage
    assert lineage is not None
    assert lineage.operation == "build_cog"
    assert lineage.params["blocksize"] == 512
    assert lineage.params["compress"] == "lzw"
    assert lineage.params["overview_resampling"] == "bilinear"
    assert art.id in lineage.inputs


# ---------------------------------------------------------------------------
# 15. Output artifact metadata fresh from file
# ---------------------------------------------------------------------------


def test_output_metadata_fresh(op, workspace):
    """Output artifact metadata comes from actual file, not input."""
    data = np.ones((512, 512), dtype=np.float32)
    raster_path = workspace / "input.tif"
    _write_raster(raster_path, data)
    art = _make_raster_artifact(raster_path)

    params = BuildCOGParams(output_path=str(workspace / "output.tif"))
    result = op.execute([art], params)

    out_art = result.artifact
    assert out_art.backing.kind == BackingStoreKind.LOCAL_FILE
    assert Path(out_art.backing.uri).exists()
    assert out_art.backing.size_bytes > 0
    assert out_art.backing.content_hash is not None
    assert out_art.spatial.crs is not None
    assert out_art.spatial.extent is not None
    assert out_art.spatial.band_count == 1
    assert out_art.metadata["format"] == "cog"
    assert result.timing_seconds > 0


# ---------------------------------------------------------------------------
# 16. All checks pass on happy path
# ---------------------------------------------------------------------------


def test_all_checks_pass_happy_path(op, workspace):
    """All declared checks pass on a well-formed run."""
    data = np.ones((512, 512), dtype=np.float32)
    raster_path = workspace / "input.tif"
    _write_raster(raster_path, data, nodata=-9999)
    art = _make_raster_artifact(raster_path)

    params = BuildCOGParams(output_path=str(workspace / "output.tif"))
    result = op.execute([art], params)

    for check in result.checks:
        assert check.state == ValidationState.VALID, (
            f"Check {check.check_name} not VALID: {check.message}"
        )


# ---------------------------------------------------------------------------
# 17. Unsupported compression rejected
# ---------------------------------------------------------------------------


def test_unsupported_compression_rejected(op, workspace):
    """Invalid compression option rejected at validation."""
    art = Artifact(
        type=ArtifactType.RASTER,
        backing=BackingStore(kind=BackingStoreKind.LOCAL_FILE, uri="/fake"),
        spatial=SpatialDescriptor(crs="EPSG:32610"),
    )
    params = BuildCOGParams(output_path="/fake/out.tif", compress="jpeg2000")
    errors = op.validate_inputs([art], params)
    assert any("Unsupported compress" in e for e in errors)


# ---------------------------------------------------------------------------
# 18. Validation: wrong input type
# ---------------------------------------------------------------------------


def test_validate_wrong_type(op, workspace):
    """Validation rejects non-raster input."""
    vector_art = Artifact(
        type=ArtifactType.VECTOR,
        backing=BackingStore(kind=BackingStoreKind.LOCAL_FILE, uri="/fake"),
        spatial=SpatialDescriptor(crs="EPSG:32610"),
    )
    params = BuildCOGParams(output_path="/fake/out.tif")
    errors = op.validate_inputs([vector_art], params)
    assert any("raster" in e.lower() for e in errors)


# ---------------------------------------------------------------------------
# 19. Validation: wrong input count
# ---------------------------------------------------------------------------


def test_validate_wrong_input_count(op, workspace):
    """Validation rejects 0 or 2 inputs."""
    params = BuildCOGParams(output_path=str(workspace / "out.tif"))
    errors = op.validate_inputs([], params)
    assert any("Exactly 1" in e for e in errors)


# ---------------------------------------------------------------------------
# 20. No-nodata raster — nodata_preserved still passes
# ---------------------------------------------------------------------------


def test_no_nodata_preserved(op, workspace):
    """Raster with no nodata → output also has no nodata → check passes."""
    data = np.ones((512, 512), dtype=np.float32)
    raster_path = workspace / "input.tif"
    _write_raster(raster_path, data)  # no nodata set
    art = _make_raster_artifact(raster_path)

    params = BuildCOGParams(output_path=str(workspace / "output.tif"))
    result = op.execute([art], params)

    with rasterio.open(workspace / "output.tif") as src:
        assert src.nodata is None

    nodata_check = [c for c in result.checks if c.check_name == "nodata_preserved"]
    assert nodata_check[0].state == ValidationState.VALID
