"""AspectOperator — compute surface aspect from DEM.

Terrain analysis operator. Calculates compass direction that each cell
faces (downslope direction). Standard GIS aspect measure.

Accepts: one raster artifact (single-band DEM)
Produces: one raster artifact (single-band aspect in degrees)
Checks: valid_range, nodata_preserved, resolution_consistent

Aspect values:
- 0° = North (slope faces north, i.e., downslope is toward north)
- 90° = East
- 180° = South
- 270° = West
- -1 (or nodata) = flat (no downslope direction)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

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
class AspectParams(OperatorParams):
    """Parameters for aspect calculation."""

    output_path: str = ""
    # Nodata value override (None = read from source)
    nodata: float | None = None
    # Nodata value for output aspect raster
    output_nodata: float = -9999.0
    # Value to use for flat areas (no downslope direction)
    flat_value: float = -1.0
    # Output convention: "compass" (0=N, 90=E, 180=S, 270=W) or
    # "math" (0=E, 90=N, 180=W, 270=S, CCW from east)
    convention: Literal["compass", "math"] = "compass"


class AspectOperator:
    """Compute surface aspect from a DEM.

    Uses central difference gradient estimation via numpy.gradient.
    Aspect is the compass direction that a slope faces (downslope direction).

    Standard GIS convention (compass):
    - 0° = North facing slope (downslope is north)
    - 90° = East facing slope
    - 180° = South facing slope
    - 270° = West facing slope
    """

    @property
    def name(self) -> str:
        return "aspect"

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
        errors = []

        if not inputs:
            errors.append("Exactly one input raster required")
            return errors

        if len(inputs) > 1:
            errors.append(f"Expected 1 input, got {len(inputs)}")

        artifact = inputs[0]

        if artifact.type != ArtifactType.RASTER:
            errors.append(f"Input must be raster, got {artifact.type.value}")

        if not artifact.is_materialized:
            errors.append("Input raster is not materialized (lazy handle)")

        if not isinstance(params, AspectParams):
            errors.append("Params must be AspectParams")
            return errors

        if not params.output_path:
            errors.append("output_path is required")

        if params.convention not in ("compass", "math"):
            errors.append(f"Invalid convention: {params.convention}")

        return errors

    def execute(self, inputs: list[Artifact], params: OperatorParams) -> OperatorResult:
        if not isinstance(params, AspectParams):
            raise OperatorError(self.name, "Params must be AspectParams")

        import rasterio

        artifact = inputs[0]
        input_path = artifact.backing.uri
        output_path = Path(params.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with rasterio.open(input_path) as src:
                if src.count != 1:
                    raise OperatorError(
                        self.name,
                        f"DEM must be single-band, got {src.count} bands",
                        inputs=[artifact.id],
                    )

                dem = src.read(1).astype(np.float64)
                nodata = params.nodata if params.nodata is not None else src.nodata
                meta = src.meta.copy()
                transform = src.transform

            # Build validity mask (True = valid cell)
            if nodata is not None:
                valid = ~np.isnan(dem) & (dem != nodata)
            else:
                valid = ~np.isnan(dem)

            # Get cell dimensions from transform
            cell_width = abs(transform.a)
            cell_height = abs(transform.e)

            if cell_width == 0 or cell_height == 0:
                raise OperatorError(
                    self.name,
                    "Invalid raster transform: zero cell dimension",
                    inputs=[artifact.id],
                )

            # Calculate gradients using central differences
            nrows, ncols = dem.shape
            if nrows == 1:
                dz_dx = np.gradient(dem, cell_width, axis=1)
                dz_dy = np.zeros_like(dem)
            elif ncols == 1:
                dz_dy = np.gradient(dem, cell_height, axis=0)
                dz_dx = np.zeros_like(dem)
            else:
                dz_dy, dz_dx = np.gradient(dem, cell_height, cell_width)

            # Calculate aspect
            # dz_dx = rate of change in x (east direction), positive = increasing east
            # dz_dy = rate of change in y (row direction), positive = increasing row (south)
            # For aspect, we want the downslope direction (where water flows)
            #
            # Compass convention: 0=N, 90=E, 180=S, 270=W
            #
            # Gradient (dz_dx, dz_dy) points in direction of steepest ASCENT
            # Aspect should point in direction of steepest DESCENT (opposite of gradient)
            #
            # So downslope vector = (-dz_dx, -dz_dy)
            # - where dz_dy is positive = slope goes UP to south, so downslope is NORTH
            # - where dz_dx is positive = slope goes UP to east, so downslope is WEST
            #
            # For arctan2(y, x): y is north component, x is east component
            # y = -dz_dy (positive when downslope has north component)
            # x = -dz_dx (positive when downslope has east component)
            # arctan2(-dz_dy, -dz_dx) gives aspect in math convention (0=E, CCW)
            # For compass (0=N, CW), we need to transform

            if params.convention == "compass":
                # Compass: 0=N, 90=E, 180=S, 270=W, clockwise
                # arctan2(-dz_dy, -dz_dx) gives math angle (0=E, CCW)
                # Convert: compass = 90 - math, then normalize
                aspect_rad = np.arctan2(-dz_dy, -dz_dx)
                aspect = 90.0 - np.degrees(aspect_rad)
                # Normalize to 0-360
                aspect = np.where(aspect < 0, aspect + 360, aspect)
                aspect = np.where(aspect >= 360, aspect - 360, aspect)
                # Flat areas get flat_value
                slope_magnitude = np.sqrt(dz_dx**2 + dz_dy**2)
                flat_mask = slope_magnitude == 0
                aspect = np.where(flat_mask, params.flat_value, aspect)
            else:
                # Math convention: 0=E, 90=N, 180=W, 270=S (CCW from east)
                aspect_rad = np.arctan2(-dz_dy, -dz_dx)
                aspect = np.degrees(aspect_rad)
                aspect = np.where(aspect < 0, aspect + 360, aspect)
                slope_magnitude = np.sqrt(dz_dx**2 + dz_dy**2)
                flat_mask = slope_magnitude == 0
                aspect = np.where(flat_mask, params.flat_value, aspect)

            # Apply nodata mask
            aspect[~valid] = params.output_nodata

            # Update metadata for output
            meta.update(
                {
                    "dtype": "float64",
                    "nodata": params.output_nodata,
                }
            )

            # Write output
            with rasterio.open(output_path, "w", **meta) as dst:
                dst.write(aspect, 1)

        except OperatorError:
            raise
        except Exception as e:
            raise OperatorError(
                self.name,
                f"Aspect calculation failed: {e}",
                inputs=[artifact.id],
            ) from e

        # Build output artifact with fresh metadata
        with rasterio.open(output_path) as out_src:
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
                    inputs=(artifact.id,),
                    params={
                        "convention": params.convention,
                        "nodata": nodata,
                        "output_nodata": params.output_nodata,
                        "flat_value": params.flat_value,
                    },
                ),
                metadata={
                    "driver": out_src.driver,
                    "dtype": str(out_src.dtypes[0]),
                    "convention": params.convention,
                    "algorithm": "central_difference_gradient",
                    "flat_value": params.flat_value,
                },
            )

        checks = self._run_checks(output_artifact, aspect, valid, params, flat_mask)
        return OperatorResult(artifact=output_artifact, checks=checks)

    def declared_checks(self) -> list[str]:
        return ["valid_range", "nodata_preserved", "resolution_consistent"]

    def _run_checks(
        self,
        output: Artifact,
        aspect: np.ndarray,
        valid: np.ndarray,
        params: AspectParams,
        flat_mask: np.ndarray,
    ) -> list[CheckResult]:
        results = []

        # Valid range check
        valid_aspect = aspect[valid & ~flat_mask]
        if len(valid_aspect) > 0:
            min_val = valid_aspect.min()
            max_val = valid_aspect.max()

            if params.convention == "compass":
                # Compass: 0-360 (flat areas excluded)
                if min_val >= 0 and max_val < 360:
                    results.append(
                        CheckResult(
                            check_name="valid_range",
                            state=ValidationState.VALID,
                            message=f"Aspect range: {min_val:.1f}° to {max_val:.1f}° (compass)",
                        )
                    )
                else:
                    results.append(
                        CheckResult(
                            check_name="valid_range",
                            state=ValidationState.INVALID,
                            message=f"Aspect out of range: {min_val:.1f}° to {max_val:.1f}°",
                        )
                    )
            else:
                # Math convention: also 0-360
                if min_val >= 0 and max_val < 360:
                    results.append(
                        CheckResult(
                            check_name="valid_range",
                            state=ValidationState.VALID,
                            message=f"Aspect range: {min_val:.1f}° to {max_val:.1f}° (math)",
                        )
                    )
                else:
                    results.append(
                        CheckResult(
                            check_name="valid_range",
                            state=ValidationState.INVALID,
                            message=f"Aspect out of range: {min_val:.1f}° to {max_val:.1f}°",
                        )
                    )
        else:
            if np.any(valid):
                # All valid cells are flat — this is valid
                results.append(
                    CheckResult(
                        check_name="valid_range",
                        state=ValidationState.VALID,
                        message="All valid cells are flat (no aspect defined)",
                    )
                )
            else:
                results.append(
                    CheckResult(
                        check_name="valid_range",
                        state=ValidationState.INVALID,
                        message="No valid aspect pixels",
                    )
                )

        # Nodata preserved check
        output_nodata_mask = aspect == params.output_nodata
        input_nodata_count = (~valid).sum()
        output_nodata_count = output_nodata_mask.sum()

        if input_nodata_count == output_nodata_count:
            results.append(
                CheckResult(
                    check_name="nodata_preserved",
                    state=ValidationState.VALID,
                    message=f"Nodata preserved: {output_nodata_count} pixels",
                )
            )
        else:
            results.append(
                CheckResult(
                    check_name="nodata_preserved",
                    state=ValidationState.WARNING,
                    message=f"Nodata mismatch: input={input_nodata_count}, output={output_nodata_count}",
                )
            )

        # Resolution consistency check
        if output.spatial and output.spatial.resolution:
            res_x, res_y = output.spatial.resolution
            if res_x > 0 and res_y > 0:
                results.append(
                    CheckResult(
                        check_name="resolution_consistent",
                        state=ValidationState.VALID,
                        message=f"Resolution: {res_x:.2f} x {res_y:.2f}",
                    )
                )
            else:
                results.append(
                    CheckResult(
                        check_name="resolution_consistent",
                        state=ValidationState.INVALID,
                        message="Invalid resolution values",
                    )
                )
        else:
            results.append(
                CheckResult(
                    check_name="resolution_consistent",
                    state=ValidationState.INVALID,
                    message="Missing spatial resolution",
                )
            )

        return results
