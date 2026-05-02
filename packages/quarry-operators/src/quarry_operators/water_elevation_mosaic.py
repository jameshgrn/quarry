"""WaterElevationMosaicOperator — combines PIXC heights with fof-compiler water mask.

Lane: operator

The fof-compiler water_frequency IS the filter. No PIXC classification
or water_frac filtering — we build our own mask from the multi-source
water occurrence stack (JRC, SARL, Dynamic World, OPERA DSWx, etc.).

Pipeline:
1. fof water_frequency > threshold → spatial water mask
2. For each PIXC pass: extract height at all pixels, mask by water mask
3. Aggregate across passes (median/mean/max) → monthly WSE

Input: 2+ artifacts (first = fof stack, rest = PIXC rasters)
Output: RASTER with 3 bands: wse, confidence (count), mask (binary)
"""

from __future__ import annotations

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
class WaterElevationMosaicParams(OperatorParams):
    """Parameters for water elevation mosaic operation."""

    output_path: str
    water_freq_threshold: float = 0.3
    aggregation: str = "median"  # median, mean, max
    fof_band_index: int = 1  # which band in fof artifact is water_frequency


class WaterElevationMosaicOperator:
    """Combines PIXC rasters with fof-compiler water frequency for WSE mosaic.

    Input: 2+ artifacts (first = fof stack, rest = PIXC rasters)
    Output: RASTER with CRS=EPSG:4326, 3 bands (wse, confidence, mask)
    """

    @property
    def name(self) -> str:
        return "water_elevation_mosaic"

    @property
    def spec(self) -> OperatorSpec:
        return OperatorSpec(
            input_types=(ArtifactType.RASTER,),
            output_type=ArtifactType.RASTER,
            min_inputs=2,
            max_inputs=-1,  # unbounded
            resource_scale=ResourceScale.HEAVY,
        )

    def validate_inputs(self, inputs: list[Artifact], params: OperatorParams) -> list[str]:
        errors = []

        if len(inputs) < 2:
            errors.append(f"Expected at least 2 inputs (fof stack + PIXC), got {len(inputs)}")
            return errors

        # First input must be fof stack
        fof_artifact = inputs[0]
        if not fof_artifact.is_materialized:
            errors.append("FOF stack artifact is not materialized (lazy handle)")

        if fof_artifact.spatial.crs is None:
            errors.append("FOF stack artifact has no CRS")

        # Check PIXC artifacts
        for i, pixc in enumerate(inputs[1:], 1):
            if not pixc.is_materialized:
                errors.append(f"PIXC artifact {i} is not materialized (lazy handle)")
            if pixc.spatial.crs is None:
                errors.append(f"PIXC artifact {i} has no CRS")

        # Params check
        if not isinstance(params, WaterElevationMosaicParams):
            errors.append("Params must be WaterElevationMosaicParams")
            return errors

        if not params.output_path:
            errors.append("output_path is required")

        # Validate aggregation method
        valid_agg = {"median", "mean", "max"}
        if params.aggregation not in valid_agg:
            errors.append(f"aggregation must be one of {valid_agg}, got {params.aggregation}")

        # Validate thresholds
        if not 0 <= params.water_freq_threshold <= 1:
            errors.append(f"water_freq_threshold must be 0-1, got {params.water_freq_threshold}")

        return errors

    def execute(self, inputs: list[Artifact], params: OperatorParams) -> OperatorResult:
        if not isinstance(params, WaterElevationMosaicParams):
            raise OperatorError(self.name, "Params must be WaterElevationMosaicParams")

        fof_artifact = inputs[0]
        pixc_artifacts = inputs[1:]

        return self._execute_mosaic(fof_artifact, pixc_artifacts, params)

    def declared_checks(self) -> list[str]:
        return [
            "extent_sane",
            "crs_valid",
            "min_observations",
            "backing_accessible",
        ]

    # -----------------------------------------------------------------------
    # Execution
    # -----------------------------------------------------------------------

    def _execute_mosaic(
        self,
        fof_artifact: Artifact,
        pixc_artifacts: list[Artifact],
        params: WaterElevationMosaicParams,
    ) -> OperatorResult:
        """Execute the mosaic operation."""
        import rasterio
        from rasterio.transform import from_bounds

        # Read FOF water frequency band
        fof_water_freq = self._read_fof_water_frequency(fof_artifact, params.fof_band_index)

        # Create spatial prior mask
        spatial_mask = fof_water_freq >= params.water_freq_threshold

        # Get output grid dimensions from FOF
        height, width = fof_water_freq.shape

        # Get extent from FOF artifact
        if fof_artifact.spatial.extent:
            xmin, ymin, xmax, ymax = fof_artifact.spatial.extent
        else:
            # Default extent
            xmin, ymin, xmax, ymax = -180, -90, 180, 90

        # Stack for accumulating height observations per pixel
        height_stack = np.full((height, width, len(pixc_artifacts)), np.nan, dtype=np.float32)

        # Process each PIXC artifact — extract ALL heights, mask by fof water
        for idx, pixc in enumerate(pixc_artifacts):
            heights = self._extract_heights(pixc, spatial_mask, height, width)
            height_stack[:, :, idx] = heights

        # Aggregate across passes → sparse WSE at PIXC-observed pixels
        sparse_wse, confidence = self._aggregate_heights(height_stack, params.aggregation)

        # Interpolate PIXC heights across the full fof water mask.
        # PIXC gives us seed elevations at sparse points; fof mask gives us
        # the dense water shape. Fill the mask with interpolated heights.
        wse = self._fill_water_mask(sparse_wse, spatial_mask)

        # The output mask is the fof water mask (dense, continuous)
        mask = spatial_mask.astype(np.uint8)

        # Write output
        output_path = Path(params.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        transform = from_bounds(xmin, ymin, xmax, ymax, width, height)

        profile = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": 3,
            "dtype": "float32",
            "crs": "EPSG:4326",
            "transform": transform,
            "compress": "deflate",
            "nodata": np.nan,
        }

        with rasterio.open(str(output_path), "w", **profile) as dst:
            dst.write(wse.astype(np.float32), 1)
            dst.write(confidence.astype(np.float32), 2)
            dst.write(mask.astype(np.float32), 3)

            dst.set_band_description(1, "wse")
            dst.set_band_description(2, "confidence")
            dst.set_band_description(3, "mask")

        # Create output artifact
        all_inputs = [fof_artifact] + pixc_artifacts
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
                crs="EPSG:4326",
                extent=(xmin, ymin, xmax, ymax),
                resolution=fof_artifact.spatial.resolution,
                band_count=3,
            ),
            lineage=Lineage(
                operation=self.name,
                inputs=tuple(a.id for a in all_inputs),
                params={
                    "water_freq_threshold": params.water_freq_threshold,
                    "aggregation": params.aggregation,
                    "num_pixc_inputs": len(pixc_artifacts),
                },
            ),
            metadata={
                "bands": ["wse", "confidence", "mask"],
                "aggregation": params.aggregation,
                "water_freq_threshold": params.water_freq_threshold,
            },
        )

        checks = self._run_checks(output_artifact, params)
        return OperatorResult(artifact=output_artifact, checks=checks)

    def _read_fof_water_frequency(self, fof_artifact: Artifact, band_index: int) -> np.ndarray:
        """Read water frequency band from FOF stack artifact."""
        import rasterio

        path = fof_artifact.backing.uri

        try:
            with rasterio.open(path) as src:
                # band_index is 1-based
                if band_index < 1 or band_index > src.count:
                    raise OperatorError(
                        self.name,
                        f"FOF band_index {band_index} out of range (1-{src.count})",
                    )
                data = src.read(band_index)
                return data.astype(np.float32)
        except Exception as e:
            raise OperatorError(self.name, f"Failed to read FOF water frequency: {e}") from e

    def _extract_heights(
        self,
        pixc_artifact: Artifact,
        spatial_mask: np.ndarray,
        target_height: int,
        target_width: int,
    ) -> np.ndarray:
        """Extract height values from PIXC raster, masked by fof water prior.

        No PIXC classification or water_frac filtering — the fof water_frequency
        IS the filter. We just read heights and mask by the spatial prior.
        """
        import rasterio

        path = pixc_artifact.backing.uri

        try:
            with rasterio.open(path) as src:
                if src.count < 2:
                    raise OperatorError(
                        self.name,
                        f"PIXC raster has {src.count} bands, need at least 2 (height is band 2)",
                    )
                heights = src.read(2).astype(np.float32)

        except Exception as e:
            raise OperatorError(self.name, f"Failed to read PIXC raster: {e}") from e

        # Resample to FOF grid if shapes differ
        if heights.shape != (target_height, target_width):
            heights = self._resample_to_grid(heights, target_height, target_width)

        # Apply fof water mask — this IS the filter
        return np.where(spatial_mask, heights, np.nan)

    def _resample_to_grid(
        self,
        data: np.ndarray,
        target_height: int,
        target_width: int,
    ) -> np.ndarray:
        """Resample data to target grid size using numpy nearest-neighbor."""
        src_h, src_w = data.shape
        row_idx = (np.arange(target_height) * src_h / target_height).astype(int)
        col_idx = (np.arange(target_width) * src_w / target_width).astype(int)
        row_idx = np.clip(row_idx, 0, src_h - 1)
        col_idx = np.clip(col_idx, 0, src_w - 1)
        return data[np.ix_(row_idx, col_idx)].astype(np.float32)

    def _fill_water_mask(self, sparse_wse: np.ndarray, water_mask: np.ndarray) -> np.ndarray:
        """Fill the fof water mask with nearest-neighbor PIXC heights.

        Vectorized iterative dilation: each round, unfilled water pixels
        absorb the mean of their filled 4-neighbors. Pure numpy.
        """
        has_value = np.isfinite(sparse_wse) & water_mask
        if not np.any(has_value):
            return np.where(water_mask, sparse_wse, np.nan)

        filled = np.where(has_value, sparse_wse, np.nan)
        unfilled = water_mask & ~has_value
        if not np.any(unfilled):
            return np.where(water_mask, filled, np.nan)

        h, w = filled.shape
        max_iters = max(h, w)  # Worst case: single seed fills entire grid

        for _ in range(max_iters):
            # Shift filled values in 4 directions, take nanmean of neighbors
            pad = np.full((h, w), np.nan, dtype=np.float32)
            neighbors = np.stack(
                [
                    np.vstack([filled[1:, :], pad[:1, :]]),  # up
                    np.vstack([pad[:1, :], filled[:-1, :]]),  # down
                    np.hstack([filled[:, 1:], pad[:, :1]]),  # left
                    np.hstack([pad[:, :1], filled[:, :-1]]),  # right
                ]
            )

            with np.errstate(all="ignore"):
                neighbor_mean = np.nanmean(neighbors, axis=0)

            # Fill unfilled water pixels that have at least one filled neighbor
            can_fill = unfilled & np.isfinite(neighbor_mean)
            if not np.any(can_fill):
                break

            filled[can_fill] = neighbor_mean[can_fill]
            unfilled = unfilled & ~can_fill

            if not np.any(unfilled):
                break

        return np.where(water_mask, filled, np.nan).astype(np.float32)

    def _aggregate_heights(
        self, height_stack: np.ndarray, aggregation: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """Aggregate height observations across passes.

        Returns:
            (wse, confidence) where confidence is count of valid observations
        """
        # Count valid observations per pixel
        valid_mask = np.isfinite(height_stack)
        confidence = valid_mask.sum(axis=2).astype(np.float32)

        if aggregation == "median":
            # Median across passes (ignoring NaN)
            wse = np.nanmedian(height_stack, axis=2)
        elif aggregation == "mean":
            # Mean across passes
            wse = np.nanmean(height_stack, axis=2)
        elif aggregation == "max":
            # Max across passes
            wse = np.nanmax(height_stack, axis=2)
        else:
            raise OperatorError(self.name, f"Unknown aggregation: {aggregation}")

        return wse.astype(np.float32), confidence

    # -----------------------------------------------------------------------
    # Checks
    # -----------------------------------------------------------------------

    def _run_checks(
        self, output: Artifact, params: WaterElevationMosaicParams
    ) -> list[CheckResult]:
        results = []

        # Extent sane
        if output.spatial.extent:
            xmin, ymin, xmax, ymax = output.spatial.extent
            if xmin < xmax and ymin < ymax:
                results.append(
                    CheckResult(
                        check_name="extent_sane",
                        state=ValidationState.VALID,
                        message=f"Extent: ({xmin:.4f}, {ymin:.4f}, {xmax:.4f}, {ymax:.4f})",
                    )
                )
            else:
                results.append(
                    CheckResult(
                        check_name="extent_sane",
                        state=ValidationState.INVALID,
                        message=f"Degenerate extent: ({xmin}, {ymin}, {xmax}, {ymax})",
                    )
                )

        # CRS valid
        if output.spatial.crs:
            results.append(
                CheckResult(
                    check_name="crs_valid",
                    state=ValidationState.VALID,
                    message=f"CRS: {output.spatial.crs}",
                )
            )
        else:
            results.append(
                CheckResult(
                    check_name="crs_valid",
                    state=ValidationState.INVALID,
                    message="Output has no CRS",
                )
            )

        # Min observations check (read back output to check)
        try:
            import rasterio

            with rasterio.open(output.backing.uri) as src:
                confidence = src.read(2)
                valid_pixels = np.sum(confidence > 0)
                total_pixels = confidence.size
                coverage = valid_pixels / total_pixels if total_pixels > 0 else 0

                if coverage > 0:
                    results.append(
                        CheckResult(
                            check_name="min_observations",
                            state=ValidationState.VALID,
                            message=f"Coverage: {coverage:.2%} ({valid_pixels} pixels)",
                        )
                    )
                else:
                    results.append(
                        CheckResult(
                            check_name="min_observations",
                            state=ValidationState.WARN,
                            message="No valid observations in output",
                        )
                    )
        except Exception as e:
            results.append(
                CheckResult(
                    check_name="min_observations",
                    state=ValidationState.WARN,
                    message=f"Could not check observations: {e}",
                )
            )

        # Backing accessible
        if output.backing and Path(output.backing.uri).exists():
            results.append(
                CheckResult(
                    check_name="backing_accessible",
                    state=ValidationState.VALID,
                    message=f"File exists: {output.backing.uri}",
                )
            )
        else:
            results.append(
                CheckResult(
                    check_name="backing_accessible",
                    state=ValidationState.INVALID,
                    message="Output file not found",
                )
            )

        return results
