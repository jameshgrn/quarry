"""SLC calibration operator pressure test.

Lane: operator

Stress points:
1. Sigma0 computation: (|SLC|^2 - noise) / xfactor
2. Zero-padded lines masked correctly
3. Multi-look NaN-aware block averaging
4. Complex multi-look preserves wrapped phase
5. Normalized interferogram magnitude bounded [0, 1]
6. Operator protocol compliance (spec, validate_inputs, declared_checks)
7. End-to-end: GeoTIFF in → calibrated GeoTIFF out
8. Invalid xfactor/noise values masked
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from quarry_core.artifact import (
    Artifact,
    ArtifactType,
    BackingStore,
    BackingStoreKind,
)
from quarry_core.operator import Operator, ResourceScale
from quarry_operators.slc_calibration import (
    SLCCalibrationOperator,
    SLCCalibrationParams,
)
from rasterio.transform import from_origin

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_raster(path: Path, data: np.ndarray, bands: int = 1) -> Artifact:
    """Write array to GeoTIFF and return an Artifact pointing to it."""
    if data.ndim == 2:
        height, width = data.shape
    else:
        bands, height, width = data.shape

    transform = from_origin(0, height, 1, 1)
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": data.shape[0] if data.ndim == 3 else 1,
        "dtype": str(data.dtype),
        "transform": transform,
        "nodata": np.nan if np.issubdtype(data.dtype, np.floating) else None,
    }

    with rasterio.open(str(path), "w", **profile) as dst:
        if data.ndim == 3:
            for i in range(data.shape[0]):
                dst.write(data[i], i + 1)
        else:
            dst.write(data, 1)

    return Artifact(
        type=ArtifactType.RASTER,
        name=path.stem,
        backing=BackingStore(
            kind=BackingStoreKind.LOCAL_FILE,
            uri=str(path),
            size_bytes=path.stat().st_size,
        ),
    )


def _make_slc_artifact(tmp_path: Path, real: np.ndarray, imag: np.ndarray) -> Artifact:
    """Create a 2-band (real, imag) raster artifact."""
    data = np.stack([real, imag], axis=0).astype(np.float32)
    return _write_raster(tmp_path / "slc.tif", data)


def _make_xfactor_artifact(tmp_path: Path, xfactor: np.ndarray) -> Artifact:
    """Create a 1-band xfactor raster artifact."""
    return _write_raster(tmp_path / "xfactor.tif", xfactor.astype(np.float32))


def _make_noise_artifact(tmp_path: Path, noise: np.ndarray) -> Artifact:
    """Create a 1-band noise raster artifact."""
    return _write_raster(tmp_path / "noise.tif", noise.astype(np.float32))


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


def test_name():
    assert SLCCalibrationOperator().name == "slc_calibrate"


def test_spec():
    op = SLCCalibrationOperator()
    assert op.spec.min_inputs == 3
    assert op.spec.max_inputs == 3
    assert op.spec.output_type == ArtifactType.RASTER
    assert op.spec.resource_scale == ResourceScale.HEAVY


def test_satisfies_operator_protocol():
    assert isinstance(SLCCalibrationOperator(), Operator)


def test_declared_checks():
    op = SLCCalibrationOperator()
    checks = op.declared_checks()
    assert "sigma0_finite" in checks
    assert "sigma0_nonnegative" in checks


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_validate_inputs_wrong_count(tmp_path: Path):
    op = SLCCalibrationOperator()
    art = _write_raster(tmp_path / "dummy.tif", np.zeros((4, 4), dtype=np.float32))
    params = SLCCalibrationParams(output_path=str(tmp_path / "out.tif"))

    errors = op.validate_inputs([art], params)
    assert any("3 inputs" in e for e in errors)


def test_validate_inputs_missing_output_path(tmp_path: Path):
    op = SLCCalibrationOperator()
    art = _write_raster(tmp_path / "dummy.tif", np.zeros((4, 4), dtype=np.float32))
    params = SLCCalibrationParams()

    errors = op.validate_inputs([art, art, art], params)
    assert any("output_path" in e for e in errors)


def test_validate_inputs_bad_looks(tmp_path: Path):
    op = SLCCalibrationOperator()
    art = _write_raster(tmp_path / "dummy.tif", np.zeros((4, 4), dtype=np.float32))
    params = SLCCalibrationParams(output_path=str(tmp_path / "out.tif"), az_looks=0)

    errors = op.validate_inputs([art, art, art], params)
    assert any("az_looks" in e for e in errors)


# ---------------------------------------------------------------------------
# Sigma0 computation (static method tests)
# ---------------------------------------------------------------------------


def test_compute_sigma0_basic():
    """sigma0 = (|SLC|^2 - noise) / xfactor for valid pixels."""
    slc = np.array(
        [[1.0 + 1.0j, 2.0 + 0.0j], [3.0 + 0.0j, 1.0 + 0.0j]],
        dtype=np.complex64,
    )
    xfactor = np.ones((2, 2), dtype=np.float32)
    noise = np.array([0.5, 0.5], dtype=np.float32)

    sigma0 = SLCCalibrationOperator._compute_sigma0(slc, xfactor, noise)

    # Row 0: |1+1j|^2=2, |2+0j|^2=4 → (2-0.5)/1=1.5, (4-0.5)/1=3.5
    np.testing.assert_allclose(sigma0[0], [1.5, 3.5])
    # Row 1: |3|^2=9, |1|^2=1 → (9-0.5)/1=8.5, (1-0.5)/1=0.5
    np.testing.assert_allclose(sigma0[1], [8.5, 0.5])


def test_compute_sigma0_masks_zero_padded_lines():
    """Lines where all SLC values are zero should be NaN."""
    slc = np.array(
        [[1.0 + 1.0j, 2.0 + 0.0j], [0.0 + 0.0j, 0.0 + 0.0j]],
        dtype=np.complex64,
    )
    xfactor = np.ones((2, 2), dtype=np.float32)
    noise = np.array([0.5, 0.0], dtype=np.float32)

    sigma0 = SLCCalibrationOperator._compute_sigma0(slc, xfactor, noise)

    np.testing.assert_allclose(sigma0[0], [1.5, 3.5])
    assert np.all(np.isnan(sigma0[1]))


def test_compute_sigma0_masks_invalid_xfactor():
    """xfactor=0 or xfactor=NaN should produce NaN sigma0."""
    slc = np.array([[1.0 + 0.0j, 1.0 + 0.0j]], dtype=np.complex64)
    xfactor = np.array([[0.0, np.nan]], dtype=np.float32)
    noise = np.array([0.0], dtype=np.float32)

    sigma0 = SLCCalibrationOperator._compute_sigma0(slc, xfactor, noise)
    assert np.all(np.isnan(sigma0[0]))


# ---------------------------------------------------------------------------
# Multi-look
# ---------------------------------------------------------------------------


def test_multilook_ignores_nan():
    data = np.array([[1.0, np.nan], [3.0, np.nan]], dtype=np.float32)
    out = SLCCalibrationOperator._multilook(data, 2, 2)
    np.testing.assert_allclose(out, [[2.0]])


def test_multilook_preserves_mean():
    data = np.array(
        [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
        dtype=np.float32,
    )
    out = SLCCalibrationOperator._multilook(data, 2, 2)
    np.testing.assert_allclose(out, [[3.5, 5.5]])


def test_complex_multilook_preserves_wrapped_phase():
    phase = np.pi - 0.1
    data = np.array(
        [[np.exp(1j * phase), np.exp(-1j * phase)]],
        dtype=np.complex64,
    )
    averaged = SLCCalibrationOperator.multilook_complex(data, 1, 2)
    averaged_phase = np.angle(averaged[0, 0])
    assert abs(abs(averaged_phase) - np.pi) < 0.01


# ---------------------------------------------------------------------------
# Normalized interferogram
# ---------------------------------------------------------------------------


def test_normalized_interferogram_magnitude_bounded():
    slc_plus = np.array([[1 + 0j, 1 + 0j]], dtype=np.complex64)
    slc_minus = np.array([[1 + 0j, -1 + 0j]], dtype=np.complex64)
    interferogram = slc_plus * np.conj(slc_minus)

    normalized = SLCCalibrationOperator.normalize_interferogram(
        interferogram, slc_plus, slc_minus, az_looks=1, rg_looks=2
    )
    np.testing.assert_allclose(np.abs(normalized), [[0.0]], atol=1e-6)


def test_normalized_interferogram_perfect_coherence():
    """Identical signals should produce coherence = 1."""
    slc = np.ones((4, 4), dtype=np.complex64) * (1 + 1j)
    interferogram = slc * np.conj(slc)

    normalized = SLCCalibrationOperator.normalize_interferogram(
        interferogram, slc, slc, az_looks=2, rg_looks=2
    )
    magnitude = np.abs(normalized)
    assert np.all(magnitude[np.isfinite(magnitude)] <= 1.0 + 1e-6)


# ---------------------------------------------------------------------------
# End-to-end execution
# ---------------------------------------------------------------------------


def test_execute_end_to_end(tmp_path: Path):
    """Full pipeline: artifacts in → calibrated sigma0 GeoTIFF out."""
    height, width = 8, 12

    # SLC with known power: |1+1j|^2 = 2
    real = np.ones((height, width), dtype=np.float32)
    imag = np.ones((height, width), dtype=np.float32)
    slc_art = _make_slc_artifact(tmp_path, real, imag)

    # xfactor = 1 everywhere
    xf_art = _make_xfactor_artifact(tmp_path, np.ones((height, width), dtype=np.float32))

    # noise = 0.5 per line
    # Noise is 2D here (written as raster), operator handles both 1D and 2D
    noise_2d = np.full((height, width), 0.5, dtype=np.float32)
    noise_art = _make_noise_artifact(tmp_path, noise_2d)

    out_path = tmp_path / "sigma0.tif"
    params = SLCCalibrationParams(
        output_path=str(out_path),
        az_looks=2,
        rg_looks=2,
    )

    op = SLCCalibrationOperator()
    result = op.execute([slc_art, xf_art, noise_art], params)

    assert result.artifact.backing.uri == str(out_path)
    assert out_path.exists()
    assert result.timing_seconds is not None
    assert result.timing_seconds > 0

    # Read output and verify sigma0 = (2 - 0.5) / 1 = 1.5
    with rasterio.open(str(out_path)) as src:
        sigma0 = src.read(1)

    expected_height = height // 2
    expected_width = width // 2
    assert sigma0.shape == (expected_height, expected_width)
    np.testing.assert_allclose(sigma0, 1.5, atol=1e-5)


def test_execute_checks_present(tmp_path: Path):
    """Checks should be populated in the result."""
    height, width = 4, 4
    real = np.ones((height, width), dtype=np.float32)
    imag = np.zeros((height, width), dtype=np.float32)
    slc_art = _make_slc_artifact(tmp_path, real, imag)
    xf_art = _make_xfactor_artifact(tmp_path, np.ones((height, width), dtype=np.float32))
    noise_art = _make_noise_artifact(tmp_path, np.zeros((height, width), dtype=np.float32))

    params = SLCCalibrationParams(
        output_path=str(tmp_path / "out.tif"),
        az_looks=1,
        rg_looks=1,
    )

    result = SLCCalibrationOperator().execute([slc_art, xf_art, noise_art], params)

    check_names = {c.check_name for c in result.checks}
    assert "sigma0_finite" in check_names
    assert "sigma0_nonnegative" in check_names


def test_execute_lineage_records_params(tmp_path: Path):
    """Lineage should record az_looks and rg_looks."""
    height, width = 4, 4
    real = np.ones((height, width), dtype=np.float32)
    imag = np.zeros((height, width), dtype=np.float32)
    slc_art = _make_slc_artifact(tmp_path, real, imag)
    xf_art = _make_xfactor_artifact(tmp_path, np.ones((height, width), dtype=np.float32))
    noise_art = _make_noise_artifact(tmp_path, np.zeros((height, width), dtype=np.float32))

    params = SLCCalibrationParams(
        output_path=str(tmp_path / "out.tif"),
        az_looks=2,
        rg_looks=3,
    )

    result = SLCCalibrationOperator().execute([slc_art, xf_art, noise_art], params)

    assert result.artifact.lineage.params["az_looks"] == 2
    assert result.artifact.lineage.params["rg_looks"] == 3
    assert len(result.artifact.lineage.inputs) == 3
