"""SlopeOperator — compute surface slope from DEM.

Lane: operator

Terrain analysis operator. Calculates slope magnitude at each cell
using elevation gradients. Standard GIS slope measure.

Accepts: one raster artifact (single-band DEM)
Produces: one raster artifact (single-band slope)
Checks: valid_range, nodata_preserved, resolution_consistent
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
class SlopeParams(OperatorParams):
    """Parameters for slope calculation."""

    output_path: str = ""
    # Output units for slope: degrees (0-90), percent (rise/run * 100),
    # radians (0-π/2), or m_m (rise/run, dimensionless)
    units: Literal["degrees", "percent", "radians", "m_m"] = "degrees"
    # Nodata value override (None = read from source)
    nodata: float | None = None
    # Nodata value for output slope raster
    output_nodata: float = -9999.0


class SlopeOperator:
    """Compute surface slope from a DEM.

    Uses central difference gradient estimation via numpy.gradient.
    Handles CRS units correctly by using actual ground resolution.
    """

    @property
    def name(self) -> str:
        return "slope"

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

        if not isinstance(params, SlopeParams):
            errors.append("Params must be SlopeParams")
            return errors

        if not params.output_path:
            errors.append("output_path is required")

        if params.units not in ("degrees", "percent", "radians", "m_m"):
            errors.append(f"Invalid units: {params.units}")

        return errors

    def execute(self, inputs: list[Artifact], params: OperatorParams) -> OperatorResult:
        if not isinstance(params, SlopeParams):
            raise OperatorError(self.name, "Params must be SlopeParams")

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
            # Transform is (a, b, c, d, e, f) where:
            # a = pixel width, e = pixel height (usually negative)
            cell_width = abs(transform.a)
            cell_height = abs(transform.e)

            if cell_width == 0 or cell_height == 0:
                raise OperatorError(
                    self.name,
                    "Invalid raster transform: zero cell dimension",
                    inputs=[artifact.id],
                )

            # Calculate gradients using central differences
            # Handle edge case: single row or single column
            nrows, ncols = dem.shape
            if nrows == 1:
                # Single row: no Y gradient, only X gradient
                dz_dx = np.gradient(dem, cell_width, axis=1)
                dz_dy = np.zeros_like(dem)
            elif ncols == 1:
                # Single column: no X gradient, only Y gradient
                dz_dy = np.gradient(dem, cell_height, axis=0)
                dz_dx = np.zeros_like(dem)
            else:
                # Normal 2D gradient
                # np.gradient returns derivatives in [y, x] order (row, col)
                dz_dy, dz_dx = np.gradient(dem, cell_height, cell_width)

            # Slope magnitude: tan(slope) = sqrt(dz_dx^2 + dz_dy^2)
            slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))

            # Convert to requested units
            if params.units == "radians":
                slope = slope_rad
            elif params.units == "degrees":
                slope = np.degrees(slope_rad)
            elif params.units == "percent":
                slope = np.tan(slope_rad) * 100.0
            else:  # m_m (rise/run, dimensionless)
                slope = np.tan(slope_rad)

            # Apply nodata mask
            slope[~valid] = params.output_nodata

            # Update metadata for output
            meta.update(
                {
                    "dtype": "float64",
                    "nodata": params.output_nodata,
                }
            )

            # Write output
            with rasterio.open(output_path, "w", **meta) as dst:
                dst.write(slope, 1)

        except OperatorError:
            raise
        except Exception as e:
            raise OperatorError(
                self.name,
                f"Slope calculation failed: {e}",
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
                        "units": params.units,
                        "nodata": nodata,
                        "output_nodata": params.output_nodata,
                    },
                ),
                metadata={
                    "driver": out_src.driver,
                    "dtype": str(out_src.dtypes[0]),
                    "units": params.units,
                    "algorithm": "central_difference_gradient",
                },
            )

        checks = self._run_checks(output_artifact, slope, valid, params)
        return OperatorResult(artifact=output_artifact, checks=checks)

    def declared_checks(self) -> list[str]:
        return ["valid_range", "nodata_preserved", "resolution_consistent"]

    def _run_checks(
        self,
        output: Artifact,
        slope: np.ndarray,
        valid: np.ndarray,
        params: SlopeParams,
    ) -> list[CheckResult]:
        results = []

        # Valid range check based on units
        valid_slope = slope[valid]
        if len(valid_slope) > 0:
            min_val = valid_slope.min()
            max_val = valid_slope.max()

            if params.units == "degrees":
                # Slope in degrees should be 0-90
                if min_val >= 0 and max_val <= 90:
                    results.append(
                        CheckResult(
                            check_name="valid_range",
                            state=ValidationState.VALID,
                            message=f"Slope range: {min_val:.2f}° to {max_val:.2f}°",
                        )
                    )
                else:
                    results.append(
                        CheckResult(
                            check_name="valid_range",
                            state=ValidationState.INVALID,
                            message=f"Slope out of range: {min_val:.2f}° to {max_val:.2f}°",
                        )
                    )
            elif params.units == "percent":
                # Percent can be anything >= 0, just check non-negative
                if min_val >= 0:
                    results.append(
                        CheckResult(
                            check_name="valid_range",
                            state=ValidationState.VALID,
                            message=f"Slope range: {min_val:.2f}% to {max_val:.2f}%",
                        )
                    )
                else:
                    results.append(
                        CheckResult(
                            check_name="valid_range",
                            state=ValidationState.INVALID,
                            message=f"Negative slope values found: {min_val:.2f}%",
                        )
                    )
            elif params.units == "radians":
                if min_val >= 0 and max_val <= np.pi / 2:
                    results.append(
                        CheckResult(
                            check_name="valid_range",
                            state=ValidationState.VALID,
                            message=f"Slope range: {min_val:.4f} to {max_val:.4f} rad",
                        )
                    )
                else:
                    results.append(
                        CheckResult(
                            check_name="valid_range",
                            state=ValidationState.INVALID,
                            message=f"Slope out of range: {min_val:.4f} to {max_val:.4f} rad",
                        )
                    )
            else:  # m_m (rise/run, dimensionless)
                # m/m can be any non-negative value (0 to theoretically infinity)
                if min_val >= 0:
                    results.append(
                        CheckResult(
                            check_name="valid_range",
                            state=ValidationState.VALID,
                            message=f"Slope range: {min_val:.4f} to {max_val:.4f} m/m",
                        )
                    )
                else:
                    results.append(
                        CheckResult(
                            check_name="valid_range",
                            state=ValidationState.INVALID,
                            message=f"Negative slope values found: {min_val:.4f} m/m",
                        )
                    )
        else:
            results.append(
                CheckResult(
                    check_name="valid_range",
                    state=ValidationState.INVALID,
                    message="No valid slope pixels",
                )
            )

        # Nodata preserved check
        output_nodata_mask = slope == params.output_nodata
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
                    state=ValidationState.WARN,
                    message=(
                        f"Nodata mismatch: input={input_nodata_count}, output={output_nodata_count}"
                    ),
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
