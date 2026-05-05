"""HillshadeOperator — compute hillshade from DEM.

Lane: operator

Terrain analysis operator. Calculates shaded relief (hillshade) from a DEM
using the Horn (1981) algorithm. Simulates illumination from a light source
at specified azimuth and altitude angles.

Accepts: one raster artifact (single-band DEM)
Produces: one raster artifact (single-band hillshade)
Checks: valid_range, nodata_preserved, resolution_consistent

Output:
- Default: uint8 raster with values 0-255 (standard hillshade convention)
- scaled=True: float64 raster with values 0.0-1.0
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
class HillshadeParams(OperatorParams):
    """Parameters for hillshade calculation."""

    output_path: str = ""
    # Sun azimuth in compass degrees (0=N, 90=E, 180=S, 270=W)
    azimuth: float = 315.0  # Default NW
    # Sun altitude in degrees above horizon (0-90)
    altitude: float = 45.0
    # Vertical exaggeration factor
    z_factor: float = 1.0
    # Nodata value override (None = read from source)
    nodata: float | None = None
    # Nodata value for uint8 output (0 is standard)
    output_nodata: float = 0.0
    # If True, output 0.0-1.0 float64 instead of 0-255 uint8
    scaled: bool = False


def _read_dem(
    input_path: str,
    op_name: str,
    artifact_id: str,
    params_nodata: float | None,
) -> tuple[np.ndarray, float | None, dict, object]:
    """Open raster, enforce single-band, return DEM and metadata.

    Returns:
        Tuple of (dem_float64, resolved_nodata, meta_copy, transform)
    """
    import rasterio

    with rasterio.open(input_path) as src:
        if src.count != 1:
            raise OperatorError(
                op_name,
                f"DEM must be single-band, got {src.count} bands",
                inputs=[artifact_id],
            )
        dem = src.read(1).astype(np.float64)
        nodata = params_nodata if params_nodata is not None else src.nodata
        meta = src.meta.copy()
        transform = src.transform
    return dem, nodata, meta, transform


def _compute_slope_aspect(
    dem: np.ndarray,
    valid: np.ndarray,
    cell_width: float,
    cell_height: float,
    z_factor: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute slope and aspect from DEM using central differences.

    Returns:
        Tuple of (slope_rad, aspect_rad)
    """
    # Mask nodata cells to NaN before scaling so gradient computation
    # produces NaN (caught later) instead of wrong values from scaled nodata
    dem_work = dem.copy()
    dem_work[~valid] = np.nan
    dem_scaled = dem_work * z_factor

    # Calculate gradients using central differences
    nrows, ncols = dem_scaled.shape
    if nrows == 1:
        dz_dx = np.gradient(dem_scaled, cell_width, axis=1)
        dz_dy = np.zeros_like(dem_scaled)
    elif ncols == 1:
        dz_dy = np.gradient(dem_scaled, cell_height, axis=0)
        dz_dx = np.zeros_like(dem_scaled)
    else:
        dz_dy, dz_dx = np.gradient(dem_scaled, cell_height, cell_width)

    # Compute slope in radians
    # tan(slope) = sqrt(dz_dx^2 + dz_dy^2)
    slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))

    # Compute aspect (downslope direction) in radians (math convention: 0=E, CCW)
    # dz_dy is d(z)/d(row) = d(z)/d(south), so the north component of
    # the downslope vector is dz_dy (not -dz_dy).
    # East component of downslope = -dz_dx.
    aspect_rad = np.arctan2(dz_dy, -dz_dx)

    return slope_rad, aspect_rad


def _compute_illumination(
    slope_rad: np.ndarray,
    aspect_rad: np.ndarray,
    valid: np.ndarray,
    azimuth_deg: float,
    altitude_deg: float,
    output_nodata: float,
) -> np.ndarray:
    """Apply Horn (1981) hillshade formula to compute illumination.

    Returns:
        Illumination array with nodata applied
    """
    # Convert sun angles
    # Zenith angle: π/2 - altitude (altitude is elevation above horizon)
    zenith_rad = np.pi / 2 - np.radians(altitude_deg)

    # Convert azimuth from compass to math convention
    # Compass: 0=N, 90=E, 180=S, 270=W (clockwise from N)
    # Math: 0=E, 90=N, 180=W, 270=S (counter-clockwise from E)
    # Conversion: math_azimuth = 90 - compass_azimuth
    azimuth_math_rad = np.radians(90.0 - azimuth_deg)

    # Horn (1981) hillshade formula
    # illumination = cos(zenith) * cos(slope) +
    #                sin(zenith) * sin(slope) * cos(azimuth - aspect)
    illumination = np.cos(zenith_rad) * np.cos(slope_rad) + np.sin(zenith_rad) * np.sin(
        slope_rad
    ) * np.cos(azimuth_math_rad - aspect_rad)

    # Clip to valid range [0, 1]
    illumination = np.clip(illumination, 0.0, 1.0)

    # Apply nodata mask — also catch NaN leaked by gradient near nodata cells
    nan_mask = np.isnan(illumination)
    illumination[~valid | nan_mask] = output_nodata

    return illumination


def _write_hillshade(
    illumination: np.ndarray,
    output_path: Path,
    meta: dict,
    scaled: bool,
    output_nodata: float,
) -> tuple[np.ndarray, str, float | int]:
    """Write hillshade raster to output path.

    Returns:
        Tuple of (hillshade_array, out_dtype, out_nodata)
    """
    import rasterio

    # Prepare output based on scaled parameter
    if scaled:
        # Output as float64 in range [0.0, 1.0]
        hillshade = illumination
        out_dtype = "float64"
        out_nodata = output_nodata
    else:
        # Output as uint8 in range [0, 255]
        hillshade = (illumination * 255).astype(np.uint8)
        out_dtype = "uint8"
        out_nodata = int(output_nodata)

    # Update metadata for output
    meta.update(
        {
            "dtype": out_dtype,
            "nodata": out_nodata,
        }
    )

    # Write output
    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(hillshade, 1)

    return hillshade, out_dtype, out_nodata


class HillshadeOperator:
    """Compute hillshade (shaded relief) from a DEM.

    Uses the Horn (1981) hillshade algorithm with central difference
    gradient estimation. Simulates illumination from a directional
    light source at specified azimuth and altitude angles.

    Formula:
        illumination = cos(zenith) * cos(slope) +
                       sin(zenith) * sin(slope) * cos(azimuth - aspect)

    Where:
        zenith = π/2 - altitude (in radians)
        azimuth is converted from compass to math convention
    """

    @property
    def name(self) -> str:
        return "hillshade"

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

        if not isinstance(params, HillshadeParams):
            errors.append("Params must be HillshadeParams")
            return errors

        if not params.output_path:
            errors.append("output_path is required")

        if not (0 <= params.azimuth <= 360):
            errors.append(f"azimuth must be 0-360, got {params.azimuth}")

        if not (0 <= params.altitude <= 90):
            errors.append(f"altitude must be 0-90, got {params.altitude}")

        if params.z_factor <= 0:
            errors.append(f"z_factor must be > 0, got {params.z_factor}")

        if not params.scaled and not (0 <= params.output_nodata <= 255):
            errors.append(
                f"output_nodata must be 0-255 for uint8 output, got {params.output_nodata}"
            )

        if not params.scaled and params.output_nodata != int(params.output_nodata):
            errors.append(
                f"output_nodata must be an integer for uint8 output, got {params.output_nodata}"
            )

        return errors

    def execute(self, inputs: list[Artifact], params: OperatorParams) -> OperatorResult:
        if not isinstance(params, HillshadeParams):
            raise OperatorError(self.name, "Params must be HillshadeParams")

        import rasterio

        artifact = inputs[0]
        input_path = artifact.backing.uri
        output_path = Path(params.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        nodata: float | None = None

        try:
            # Read DEM and metadata
            dem, nodata, meta, transform = _read_dem(
                input_path,
                self.name,
                artifact.id,
                params.nodata,
            )

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

            # Compute slope and aspect
            slope_rad, aspect_rad = _compute_slope_aspect(
                dem,
                valid,
                cell_width,
                cell_height,
                params.z_factor,
            )

            # Compute illumination using Horn (1981) formula
            illumination = _compute_illumination(
                slope_rad,
                aspect_rad,
                valid,
                params.azimuth,
                params.altitude,
                params.output_nodata,
            )

            # Write hillshade output
            hillshade, out_dtype, out_nodata = _write_hillshade(
                illumination,
                output_path,
                meta,
                params.scaled,
                params.output_nodata,
            )

        except OperatorError:
            raise
        except Exception as e:
            raise OperatorError(
                self.name,
                f"Hillshade calculation failed: {e}",
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
                        "azimuth": params.azimuth,
                        "altitude": params.altitude,
                        "z_factor": params.z_factor,
                        "scaled": params.scaled,
                        "nodata": nodata,
                        "output_nodata": params.output_nodata,
                    },
                ),
                metadata={
                    "driver": out_src.driver,
                    "dtype": str(out_src.dtypes[0]),
                    "azimuth": params.azimuth,
                    "altitude": params.altitude,
                    "z_factor": params.z_factor,
                    "algorithm": "horn_1981",
                },
            )

        checks = self._run_checks(output_artifact, hillshade, valid, params)
        return OperatorResult(artifact=output_artifact, checks=checks)

    def declared_checks(self) -> list[str]:
        return ["valid_range", "nodata_preserved", "resolution_consistent"]

    def _run_checks(
        self,
        output: Artifact,
        hillshade: np.ndarray,
        valid: np.ndarray,
        params: HillshadeParams,
    ) -> list[CheckResult]:
        results = []

        # Valid range check
        valid_hillshade = hillshade[valid]
        if len(valid_hillshade) > 0:
            min_val = valid_hillshade.min()
            max_val = valid_hillshade.max()

            if params.scaled:
                # Scaled output: 0.0-1.0
                if min_val >= 0.0 and max_val <= 1.0:
                    results.append(
                        CheckResult(
                            check_name="valid_range",
                            state=ValidationState.VALID,
                            message=f"Hillshade range: {min_val:.4f} to {max_val:.4f} (scaled)",
                        )
                    )
                else:
                    results.append(
                        CheckResult(
                            check_name="valid_range",
                            state=ValidationState.INVALID,
                            message=f"Hillshade out of range: {min_val:.4f} to {max_val:.4f}",
                        )
                    )
            else:
                # Standard output: 0-255
                if min_val >= 0 and max_val <= 255:
                    results.append(
                        CheckResult(
                            check_name="valid_range",
                            state=ValidationState.VALID,
                            message=f"Hillshade range: {int(min_val)} to {int(max_val)} (uint8)",
                        )
                    )
                else:
                    results.append(
                        CheckResult(
                            check_name="valid_range",
                            state=ValidationState.INVALID,
                            message=f"Hillshade out of range: {min_val} to {max_val}",
                        )
                    )
        else:
            results.append(
                CheckResult(
                    check_name="valid_range",
                    state=ValidationState.INVALID,
                    message="No valid hillshade pixels",
                )
            )

        # Nodata preserved check
        if params.scaled:
            output_nodata_mask = hillshade == params.output_nodata
        else:
            output_nodata_mask = hillshade == int(params.output_nodata)
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
