"""SLCCalibrationOperator — calibrate SWOT SLC to sigma0 and interferometric products.

Lane: operator

Accepts: 3 raster artifacts (SLC complex data, xfactor calibration, noise vector)
Produces: 1 raster artifact (calibrated sigma0)

Key formula: sigma0 = (|SLC|^2 - noise) / xfactor

Also produces interferometric products (magnitude, phase) when given
both plus_y and minus_y antenna pairs.

Multi-look (block averaging) is bundled here because sigma0 calibration
and interferogram normalization require it inline.

Reference: JPL D-56410 SWOT Product Description L1B HR SLC
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from quarry_core.artifact import (
    Artifact,
    ArtifactType,
    BackingStore,
    BackingStoreKind,
    CheckResult,
    Lineage,
    SpatialDescriptor,
    ValidationState,
    content_hash,
)
from quarry_core.operator import (
    OperatorError,
    OperatorParams,
    OperatorResult,
    OperatorSpec,
    ResourceScale,
)


@dataclass(frozen=True)
class SLCCalibrationParams(OperatorParams):
    """Parameters for SLC calibration."""

    output_path: str = ""
    az_looks: int = 4
    rg_looks: int = 4


class SLCCalibrationOperator:
    """Calibrate SWOT SLC complex data to sigma0.

    Inputs (3 artifacts, ordered):
        [0] SLC — complex raster (2-band: real, imag from HDF5Connector)
        [1] xfactor — calibration factor raster (1-band)
        [2] noise — noise vector raster (1-band)

    Output:
        Calibrated sigma0 raster (1-band, float32) after multi-look.
    """

    @property
    def name(self) -> str:
        return "slc_calibrate"

    @property
    def spec(self) -> OperatorSpec:
        return OperatorSpec(
            input_types=(ArtifactType.RASTER,),
            output_type=ArtifactType.RASTER,
            min_inputs=3,
            max_inputs=3,
            resource_scale=ResourceScale.HEAVY,
        )

    def validate_inputs(self, inputs: list[Artifact], params: OperatorParams) -> list[str]:
        errors: list[str] = []

        if len(inputs) != 3:
            errors.append(f"Expected 3 inputs (slc, xfactor, noise), got {len(inputs)}")
            return errors

        for i, artifact in enumerate(inputs):
            if artifact.type != ArtifactType.RASTER:
                errors.append(f"Input [{i}] must be raster, got {artifact.type.value}")
            if not artifact.is_materialized:
                errors.append(f"Input [{i}] is not materialized (lazy handle)")

        if not isinstance(params, SLCCalibrationParams):
            errors.append("Params must be SLCCalibrationParams")
            return errors

        if not params.output_path:
            errors.append("output_path is required")

        if params.az_looks < 1:
            errors.append(f"az_looks must be >= 1, got {params.az_looks}")
        if params.rg_looks < 1:
            errors.append(f"rg_looks must be >= 1, got {params.rg_looks}")

        return errors

    def execute(self, inputs: list[Artifact], params: OperatorParams) -> OperatorResult:
        if not isinstance(params, SLCCalibrationParams):
            raise OperatorError(self.name, "Params must be SLCCalibrationParams")

        import rasterio
        from rasterio.transform import from_bounds

        t0 = time.monotonic()

        slc_artifact, xfactor_artifact, noise_artifact = inputs
        output_path = Path(params.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Read SLC complex data (2-band: real, imag)
            with rasterio.open(slc_artifact.backing.uri) as src:
                real = src.read(1).astype(np.float32)
                imag = src.read(2).astype(np.float32)

            slc = self._build_complex(real, imag)

            # Read calibration factor
            with rasterio.open(xfactor_artifact.backing.uri) as src:
                xfactor = src.read(1).astype(np.float32)

            # Read noise vector
            with rasterio.open(noise_artifact.backing.uri) as src:
                noise = src.read(1).astype(np.float32)

            # Compute calibrated sigma0
            sigma0 = self._compute_sigma0(slc, xfactor, noise)

            # Multi-look
            sigma0 = self._multilook(sigma0, params.az_looks, params.rg_looks)

            # Write output
            height, width = sigma0.shape
            slc_spatial = slc_artifact.spatial
            extent = slc_spatial.extent or (0, 0, width, height)
            transform = from_bounds(extent[0], extent[1], extent[2], extent[3], width, height)

            profile = {
                "driver": "GTiff",
                "height": height,
                "width": width,
                "count": 1,
                "dtype": "float32",
                "crs": slc_spatial.crs,
                "transform": transform,
                "compress": "deflate",
                "nodata": np.nan,
            }

            with rasterio.open(str(output_path), "w", **profile) as dst:
                dst.write(sigma0, 1)
                dst.set_band_description(1, "sigma0")

        except OperatorError:
            raise
        except Exception as e:
            raise OperatorError(
                self.name,
                f"SLC calibration failed: {e}",
                inputs=[a.id for a in inputs],
            ) from e

        elapsed = time.monotonic() - t0

        # Build output artifact with fresh metadata
        with rasterio.open(str(output_path)) as out_src:
            out_bounds = out_src.bounds
            output_artifact = Artifact(
                type=ArtifactType.RASTER,
                name=output_path.stem,
                backing=BackingStore(
                    kind=BackingStoreKind.LOCAL_FILE,
                    uri=str(output_path),
                    size_bytes=output_path.stat().st_size,
                    content_hash=content_hash(output_path),
                ),
                spatial=SpatialDescriptor(
                    crs=str(out_src.crs) if out_src.crs else None,
                    extent=(
                        out_bounds.left,
                        out_bounds.bottom,
                        out_bounds.right,
                        out_bounds.top,
                    ),
                    resolution=(out_src.res[0], out_src.res[1]),
                    band_count=out_src.count,
                ),
                lineage=Lineage(
                    operation=self.name,
                    inputs=tuple(a.id for a in inputs),
                    params={
                        "az_looks": params.az_looks,
                        "rg_looks": params.rg_looks,
                    },
                ),
                metadata={
                    "product": "sigma0",
                    "algorithm": "slc_power_minus_noise_over_xfactor",
                    "az_looks": params.az_looks,
                    "rg_looks": params.rg_looks,
                },
            )

        checks = self._run_checks(sigma0, params)
        return OperatorResult(
            artifact=output_artifact,
            checks=checks,
            timing_seconds=elapsed,
        )

    def declared_checks(self) -> list[str]:
        return ["sigma0_finite", "sigma0_nonnegative"]

    # -----------------------------------------------------------------------
    # Core processing
    # -----------------------------------------------------------------------

    @staticmethod
    def _build_complex(real: np.ndarray, imag: np.ndarray) -> np.ndarray:
        """Build complex array from real/imag bands, masking invalid values."""
        invalid = (
            ~np.isfinite(real)
            | ~np.isfinite(imag)
            | (np.abs(real) >= 1e20)
            | (np.abs(imag) >= 1e20)
        )
        slc = real + 1j * imag
        slc[invalid] = np.nan + 1j * np.nan
        return slc

    @staticmethod
    def _compute_sigma0(slc: np.ndarray, xfactor: np.ndarray, noise: np.ndarray) -> np.ndarray:
        """Compute calibrated sigma0: sigma0 = (|SLC|^2 - noise) / xfactor.

        Handles zero-padded lines (all-zero rows in SLC) and invalid
        calibration values.
        """
        power = (np.abs(slc) ** 2).astype(np.float32)
        finite_power = np.isfinite(power)

        # Mask invalid: zero-padded lines + invalid xfactor
        has_signal = np.any(finite_power & (power > 0), axis=1)

        # Broadcast noise to 2D if it's 1D (per-line noise)
        if noise.ndim == 1:
            noise_2d = np.broadcast_to(noise[:, np.newaxis], power.shape)
        else:
            noise_2d = noise

        valid = (
            has_signal[:, np.newaxis]
            & finite_power
            & (xfactor != 0)
            & np.isfinite(xfactor)
            & (np.abs(xfactor) < 1e20)
            & np.isfinite(noise_2d)
            & (np.abs(noise_2d) < 1e20)
        )

        sigma0 = np.full(power.shape, np.nan, dtype=np.float32)
        sigma0[valid] = (power[valid] - noise_2d[valid]) / xfactor[valid]

        return sigma0

    @staticmethod
    def _multilook(data: np.ndarray, az_looks: int, rg_looks: int) -> np.ndarray:
        """Reduce resolution by NaN-aware block-averaging."""
        lines, pixels = data.shape
        al = (lines // az_looks) * az_looks
        rl = (pixels // rg_looks) * rg_looks
        trimmed = data[:al, :rl]
        blocks = trimmed.reshape(al // az_looks, az_looks, rl // rg_looks, rg_looks)
        valid = np.isfinite(blocks)
        counts = valid.sum(axis=(1, 3))
        sums = np.where(valid, blocks, 0).sum(axis=(1, 3), dtype=np.float64)
        out = np.full(counts.shape, np.nan, dtype=np.float32)
        np.divide(sums, counts, out=out, where=counts > 0)
        return out

    @staticmethod
    def multilook_complex(data: np.ndarray, az_looks: int, rg_looks: int) -> np.ndarray:
        """Reduce a complex raster by coherent block averaging.

        Public because InterferogramOperator (future) will need this.
        """
        lines, pixels = data.shape
        al = (lines // az_looks) * az_looks
        rl = (pixels // rg_looks) * rg_looks
        trimmed = data[:al, :rl]
        blocks = trimmed.reshape(al // az_looks, az_looks, rl // rg_looks, rg_looks)
        valid = np.isfinite(blocks.real) & np.isfinite(blocks.imag)
        counts = valid.sum(axis=(1, 3))
        sums = np.where(valid, blocks, 0).sum(axis=(1, 3), dtype=np.complex128)
        out = np.full(counts.shape, np.nan + 1j * np.nan, dtype=np.complex64)
        np.divide(sums, counts, out=out, where=counts > 0)
        return out

    @staticmethod
    def normalize_interferogram(
        interferogram: np.ndarray,
        slc_plus: np.ndarray,
        slc_minus: np.ndarray,
        az_looks: int,
        rg_looks: int,
    ) -> np.ndarray:
        """Compute multilooked normalized conjugate product.

        Public because InterferogramOperator (future) will call this.
        Returns complex array with |magnitude| <= 1.
        """
        ml = SLCCalibrationOperator.multilook_complex
        ml_real = SLCCalibrationOperator._multilook

        numerator = ml(interferogram, az_looks, rg_looks)
        plus_power = ml_real((np.abs(slc_plus) ** 2).astype(np.float32), az_looks, rg_looks)
        minus_power = ml_real((np.abs(slc_minus) ** 2).astype(np.float32), az_looks, rg_looks)

        denominator = np.sqrt(plus_power * minus_power).astype(np.float32)
        valid = np.isfinite(denominator) & (denominator > 0)

        normalized = np.full(numerator.shape, np.nan + 1j * np.nan, dtype=np.complex64)
        np.divide(numerator, denominator, out=normalized, where=valid)

        # Clip magnitude overshoots to 1.0
        magnitude = np.abs(normalized)
        overshoot = np.isfinite(magnitude) & (magnitude > 1.0)
        normalized[overshoot] /= magnitude[overshoot].astype(np.complex64)

        return normalized

    # -----------------------------------------------------------------------
    # Checks
    # -----------------------------------------------------------------------

    def _run_checks(self, sigma0: np.ndarray, params: SLCCalibrationParams) -> list[CheckResult]:
        results: list[CheckResult] = []

        valid_pixels = sigma0[np.isfinite(sigma0)]

        if len(valid_pixels) == 0:
            results.append(
                CheckResult(
                    check_name="sigma0_finite",
                    state=ValidationState.INVALID,
                    message="No finite sigma0 pixels",
                )
            )
            return results

        finite_ratio = len(valid_pixels) / sigma0.size
        results.append(
            CheckResult(
                check_name="sigma0_finite",
                state=ValidationState.VALID if finite_ratio > 0.01 else ValidationState.WARN,
                message=f"Finite pixels: {finite_ratio:.1%} ({len(valid_pixels)}/{sigma0.size})",
            )
        )

        negative_count = (valid_pixels < 0).sum()
        if negative_count == 0:
            results.append(
                CheckResult(
                    check_name="sigma0_nonnegative",
                    state=ValidationState.VALID,
                    message="All finite sigma0 values are non-negative",
                )
            )
        else:
            neg_ratio = negative_count / len(valid_pixels)
            results.append(
                CheckResult(
                    check_name="sigma0_nonnegative",
                    state=ValidationState.WARN if neg_ratio < 0.1 else ValidationState.INVALID,
                    message=(
                        f"Negative sigma0: {neg_ratio:.1%} ({negative_count}/{len(valid_pixels)})"
                    ),
                )
            )

        return results
