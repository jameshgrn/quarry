"""ClipRasterOperator — clips a raster artifact to bounds or mask.

Accepts: one raster artifact (+ optional vector mask as second input)
Produces: one raster artifact (clipped)
Checks: crs_valid, extent_within_input, backing_accessible
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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
class ClipRasterParams(OperatorParams):
    """Parameters for raster clipping."""

    # Clip by bounds (xmin, ymin, xmax, ymax) — used if no mask input
    bounds: tuple[float, float, float, float] | None = None

    # Output path (required)
    output_path: str = ""

    # Whether to crop the raster to the clip extent (vs just masking)
    crop: bool = True


class ClipRasterOperator:
    """Clips a raster to bounds or a vector mask."""

    @property
    def name(self) -> str:
        return "clip_raster"

    @property
    def spec(self) -> OperatorSpec:
        return OperatorSpec(
            input_types=(ArtifactType.RASTER, ArtifactType.VECTOR),
            output_type=ArtifactType.RASTER,
            min_inputs=1,
            max_inputs=2,  # raster + optional vector mask
            resource_scale=ResourceScale.LIGHT,
        )

    def validate_inputs(self, inputs: list[Artifact], params: OperatorParams) -> list[str]:
        """Validate that inputs and params are sane."""
        errors = []

        if not inputs:
            errors.append("At least one input artifact required")
            return errors

        # First input must be raster
        if inputs[0].type != ArtifactType.RASTER:
            errors.append(f"First input must be raster, got {inputs[0].type.value}")

        # First input must be materialized
        if not inputs[0].is_materialized:
            errors.append("Input raster is not materialized (lazy handle)")

        # If second input exists, must be vector (mask)
        if len(inputs) > 1:
            if inputs[1].type != ArtifactType.VECTOR:
                errors.append(f"Second input (mask) must be vector, got {inputs[1].type.value}")
            if not inputs[1].is_materialized:
                errors.append("Mask vector is not materialized")

        # Params check
        if not isinstance(params, ClipRasterParams):
            errors.append("Params must be ClipRasterParams")
            return errors

        # Need either bounds or a mask
        if params.bounds is None and len(inputs) < 2:
            errors.append("Either bounds or a vector mask input is required")

        if not params.output_path:
            errors.append("output_path is required")

        return errors

    def execute(self, inputs: list[Artifact], params: OperatorParams) -> OperatorResult:
        """Execute the clip operation."""
        if not isinstance(params, ClipRasterParams):
            raise OperatorError(self.name, "Params must be ClipRasterParams")

        import rasterio

        raster_artifact = inputs[0]
        raster_path = raster_artifact.backing.uri
        output_path = Path(params.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        checks: list[CheckResult] = []

        try:
            with rasterio.open(raster_path) as src:
                if len(inputs) > 1:
                    # Clip by vector mask
                    mask_path = inputs[1].backing.uri
                    out_image, out_transform = self._clip_by_mask(src, mask_path, params.crop)
                else:
                    # Clip by bounds
                    out_image, out_transform = self._clip_by_bounds(src, params.bounds, params.crop)

                # Write output
                out_meta = src.meta.copy()
                out_meta.update(
                    {
                        "height": out_image.shape[1],
                        "width": out_image.shape[2],
                        "transform": out_transform,
                    }
                )

                with rasterio.open(output_path, "w", **out_meta) as dst:
                    dst.write(out_image)

        except Exception as e:
            raise OperatorError(
                self.name,
                f"Clip failed: {e}",
                inputs=[a.id for a in inputs],
            ) from e

        # Build output artifact
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
                    extent=(out_bounds.left, out_bounds.bottom, out_bounds.right, out_bounds.top),
                    resolution=(out_src.res[0], out_src.res[1]),
                    band_count=out_src.count,
                ),
                lineage=Lineage(
                    operation=self.name,
                    inputs=tuple(a.id for a in inputs),
                    params={"bounds": params.bounds, "crop": params.crop},
                ),
                metadata={"driver": out_src.driver, "dtype": str(out_src.dtypes[0])},
            )

        # Run declared checks
        checks.extend(self._run_checks(output_artifact, inputs))

        return OperatorResult(
            artifact=output_artifact,
            checks=checks,
        )

    def declared_checks(self) -> list[str]:
        return ["crs_valid", "extent_within_input", "backing_accessible"]

    # -----------------------------------------------------------------------
    # Private
    # -----------------------------------------------------------------------

    def _clip_by_bounds(self, src, bounds, crop):
        """Clip raster by bounding box."""
        from rasterio.windows import from_bounds

        window = from_bounds(*bounds, transform=src.transform)
        out_image = src.read(window=window)
        out_transform = src.window_transform(window)
        return out_image, out_transform

    def _clip_by_mask(self, src, mask_path, crop):
        """Clip raster by vector mask geometry."""
        import fiona
        from rasterio.mask import mask as rasterio_mask
        from shapely.geometry import shape

        with fiona.open(mask_path) as mask_src:
            geometries = [shape(f["geometry"]) for f in mask_src]

        # Convert shapely geometries to GeoJSON-like dicts for rasterio
        geojson_geoms = [g.__geo_interface__ for g in geometries]
        out_image, out_transform = rasterio_mask(src, geojson_geoms, crop=crop)
        return out_image, out_transform

    def _run_checks(self, output: Artifact, inputs: list[Artifact]) -> list[CheckResult]:
        """Run declared checks on output."""
        results = []

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

        # Extent within input
        if output.spatial.extent and inputs[0].spatial.extent:
            ox = output.spatial.extent
            ix = inputs[0].spatial.extent
            within = (
                ox[0] >= ix[0] - 1e-6
                and ox[1] >= ix[1] - 1e-6
                and ox[2] <= ix[2] + 1e-6
                and ox[3] <= ix[3] + 1e-6
            )
            if within:
                results.append(
                    CheckResult(
                        check_name="extent_within_input",
                        state=ValidationState.VALID,
                        message="Output extent is within input extent",
                    )
                )
            else:
                results.append(
                    CheckResult(
                        check_name="extent_within_input",
                        state=ValidationState.WARN,
                        message="Output extent exceeds input extent",
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
