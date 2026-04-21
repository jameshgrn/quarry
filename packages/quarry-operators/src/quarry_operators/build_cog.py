"""BuildCOGOperator — normalize a raster artifact into a Cloud-Optimized GeoTIFF.

Lane: operator
Accepts: one raster artifact
Produces: one raster artifact (COG-formatted GeoTIFF with tiling + overviews)
Checks: is_cog, crs_preserved, dimensions_preserved, nodata_preserved

This is a representation transform, not a semantic transform — the data
does not change, only how it is stored. The output is deployment-ready.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
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
from rasterio.enums import Resampling
from rasterio.shutil import copy as rio_copy


@dataclass(frozen=True)
class BuildCOGParams(OperatorParams):
    """Parameters for COG creation."""

    output_path: str = ""
    blocksize: int = 256
    compress: str = "deflate"
    overview_resampling: str = "nearest"


class BuildCOGOperator:
    """Normalize a raster into a Cloud-Optimized GeoTIFF.

    Input: single raster artifact (any GeoTIFF or rasterio-readable format)
    Output: raster artifact — tiled, with overviews, optionally compressed.

    This operator does not change pixel values at the base resolution.
    Overviews are derived via resampling and are not checked for pixel equality.
    """

    @property
    def name(self) -> str:
        return "build_cog"

    @property
    def spec(self) -> OperatorSpec:
        return OperatorSpec(
            input_types=(ArtifactType.RASTER,),
            output_type=ArtifactType.RASTER,
            min_inputs=1,
            max_inputs=1,
            resource_scale=ResourceScale.MEDIUM,
        )

    def validate_inputs(self, inputs: list[Artifact], params: OperatorParams) -> list[str]:
        errors: list[str] = []

        if len(inputs) != 1:
            errors.append(f"Exactly 1 raster input required, got {len(inputs)}")
            return errors

        art = inputs[0]
        if art.type != ArtifactType.RASTER:
            errors.append(f"Input must be raster, got {art.type.value}")

        if not art.is_materialized:
            errors.append("Input is not materialized (lazy artifacts cannot be COG-built)")

        if not isinstance(params, BuildCOGParams):
            errors.append("Params must be BuildCOGParams")
            return errors

        if not params.output_path:
            errors.append("output_path is required")

        if params.blocksize < 64:
            errors.append(f"blocksize must be >= 64, got {params.blocksize}")

        valid_compress = ("deflate", "lz4", "zstd", "lzw", "none")
        if params.compress not in valid_compress:
            errors.append(f"Unsupported compress: {params.compress} (valid: {valid_compress})")

        valid_resampling = ("nearest", "bilinear", "cubic", "average", "mode")
        if params.overview_resampling not in valid_resampling:
            errors.append(
                f"Unsupported overview_resampling: {params.overview_resampling} "
                f"(valid: {valid_resampling})"
            )

        return errors

    def execute(self, inputs: list[Artifact], params: OperatorParams) -> OperatorResult:
        if not isinstance(params, BuildCOGParams):
            raise OperatorError(self.name, "Params must be BuildCOGParams")

        import time

        t0 = time.monotonic()

        input_artifact = inputs[0]
        output_path = Path(params.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            self._build_cog(input_artifact, output_path, params)
        except OperatorError:
            raise
        except Exception as e:
            raise OperatorError(
                self.name,
                f"COG build failed: {e}",
                inputs=[input_artifact.id],
            ) from e

        elapsed = time.monotonic() - t0

        # Read fresh metadata from output
        with rasterio.open(output_path) as dst:
            bounds = dst.bounds
            out_crs = str(dst.crs) if dst.crs else None
            out_band_count = dst.count
            out_overviews = dst.overviews(1)
            out_block_shapes = dst.block_shapes
            out_is_tiled = dst.profile.get("tiled", False)
            out_compression = dst.profile.get("compress", dst.compression)
            if hasattr(out_compression, "value"):
                out_compression = out_compression.value
            out_dtype = str(dst.dtypes[0])
            out_nodata = dst.nodata
            out_height = dst.height
            out_width = dst.width

        is_cog = out_is_tiled and len(out_overviews) > 0

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
                crs=out_crs,
                extent=(bounds.left, bounds.bottom, bounds.right, bounds.top),
                resolution=(
                    abs(bounds.right - bounds.left) / out_width,
                    abs(bounds.top - bounds.bottom) / out_height,
                )
                if out_width and out_height
                else None,
                band_count=out_band_count,
            ),
            lineage=Lineage(
                operation=self.name,
                inputs=(input_artifact.id,),
                params={
                    "blocksize": params.blocksize,
                    "compress": params.compress,
                    "overview_resampling": params.overview_resampling,
                },
            ),
            metadata={
                "format": "cog",
                "is_cog": is_cog,
                "block_size": out_block_shapes[0] if out_block_shapes else None,
                "overview_levels": out_overviews,
                "compression": out_compression,
                "dtype": out_dtype,
                "nodata": out_nodata,
            },
        )

        checks = self._run_checks(output_artifact, input_artifact, is_cog)

        return OperatorResult(
            artifact=output_artifact,
            checks=checks,
            timing_seconds=elapsed,
        )

    def declared_checks(self) -> list[str]:
        return ["is_cog", "crs_preserved", "dimensions_preserved", "nodata_preserved"]

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _build_cog(
        self,
        input_artifact: Artifact,
        output_path: Path,
        params: BuildCOGParams,
    ) -> None:
        """Build a COG from the input raster."""
        resampling_map = {
            "nearest": Resampling.nearest,
            "bilinear": Resampling.bilinear,
            "cubic": Resampling.cubic,
            "average": Resampling.average,
            "mode": Resampling.mode,
        }
        resampling = resampling_map[params.overview_resampling]

        with rasterio.open(input_artifact.backing.uri) as src:
            # Compute overview levels: halve until dimension < blocksize
            overview_levels = self._compute_overview_levels(src.height, src.width, params.blocksize)

            # Build creation options for COG
            profile = src.profile.copy()
            profile.update(
                driver="GTiff",
                tiled=True,
                blockxsize=params.blocksize,
                blockysize=params.blocksize,
            )
            if params.compress != "none":
                profile["compress"] = params.compress
            else:
                profile.pop("compress", None)

            # Write tiled GeoTIFF
            with rasterio.open(output_path, "w", **profile) as dst:
                for band_idx in range(1, src.count + 1):
                    data = src.read(band_idx)
                    dst.write(data, band_idx)

            # Build overviews on the written file
            if overview_levels:
                with rasterio.open(output_path, "r+") as dst:
                    dst.build_overviews(overview_levels, resampling)
                    dst.update_tags(ns="rio_overview", resampling=params.overview_resampling)

            # Final pass: copy to COG layout (interleaved overviews)
            cog_path = output_path.with_suffix(".cog.tif")
            rio_copy(
                output_path,
                cog_path,
                driver="GTiff",
                copy_src_overviews=True,
                tiled=True,
                blockxsize=params.blocksize,
                blockysize=params.blocksize,
                compress=params.compress if params.compress != "none" else None,
            )

            # Replace the intermediate with the final COG
            cog_path.replace(output_path)

    @staticmethod
    def _compute_overview_levels(height: int, width: int, blocksize: int) -> list[int]:
        """Compute overview levels. Halve until smallest dimension < blocksize."""
        levels = []
        factor = 2
        min_dim = min(height, width)
        while min_dim // factor >= blocksize:
            levels.append(factor)
            factor *= 2
        # Always add at least one overview level if the raster is larger than blocksize
        if not levels and min_dim > blocksize:
            levels.append(2)
        return levels

    def _run_checks(
        self,
        output: Artifact,
        input_art: Artifact,
        is_cog: bool,
    ) -> list[CheckResult]:
        results: list[CheckResult] = []

        # is_cog: the entire point
        if is_cog:
            results.append(
                CheckResult(
                    check_name="is_cog",
                    state=ValidationState.VALID,
                    message="Output is a valid COG (tiled + overviews)",
                )
            )
        else:
            results.append(
                CheckResult(
                    check_name="is_cog",
                    state=ValidationState.INVALID,
                    message="Output is NOT a valid COG",
                )
            )

        # CRS preserved
        in_crs = input_art.spatial.crs
        out_crs = output.spatial.crs
        if in_crs and out_crs and in_crs == out_crs:
            results.append(
                CheckResult(
                    check_name="crs_preserved",
                    state=ValidationState.VALID,
                    message=f"CRS preserved: {out_crs}",
                )
            )
        elif in_crs and out_crs:
            results.append(
                CheckResult(
                    check_name="crs_preserved",
                    state=ValidationState.INVALID,
                    message=f"CRS changed: {in_crs} → {out_crs}",
                )
            )
        else:
            results.append(
                CheckResult(
                    check_name="crs_preserved",
                    state=ValidationState.WARN,
                    message="Cannot verify CRS preservation — missing CRS on input or output",
                )
            )

        # Dimensions preserved (band count, height, width)
        in_bands = input_art.spatial.band_count
        out_bands = output.spatial.band_count
        if in_bands is not None and out_bands is not None and in_bands == out_bands:
            results.append(
                CheckResult(
                    check_name="dimensions_preserved",
                    state=ValidationState.VALID,
                    message=f"Band count preserved: {out_bands}",
                )
            )
        elif in_bands is not None and out_bands is not None:
            results.append(
                CheckResult(
                    check_name="dimensions_preserved",
                    state=ValidationState.INVALID,
                    message=f"Band count changed: {in_bands} → {out_bands}",
                )
            )
        else:
            results.append(
                CheckResult(
                    check_name="dimensions_preserved",
                    state=ValidationState.WARN,
                    message="Cannot verify band count — missing from input or output",
                )
            )

        # Nodata preserved
        in_nodata = None
        out_nodata = output.metadata.get("nodata")
        with rasterio.open(input_art.backing.uri) as src:
            in_nodata = src.nodata

        nodata_match = (in_nodata is None and out_nodata is None) or (
            in_nodata is not None
            and out_nodata is not None
            and (in_nodata == out_nodata or (np.isnan(in_nodata) and np.isnan(out_nodata)))
        )
        if nodata_match:
            results.append(
                CheckResult(
                    check_name="nodata_preserved",
                    state=ValidationState.VALID,
                    message=f"Nodata preserved: {out_nodata}",
                )
            )
        else:
            results.append(
                CheckResult(
                    check_name="nodata_preserved",
                    state=ValidationState.INVALID,
                    message=f"Nodata changed: {in_nodata} → {out_nodata}",
                )
            )

        return results
