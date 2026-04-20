"""ReprojectOperator — reprojects a raster or vector artifact to a new CRS.

Pressures:
- Spatial descriptor updates (CRS changes, extent transforms, resolution recalc)
- Output metadata regeneration (not just copying input metadata)
- Check composition (input CRS valid, output CRS valid, extent sanity post-transform)
- Lineage capturing transform intent (target CRS, resampling method)
- Lazy artifact handling (should reject — can't reproject without data)

Accepts: one raster OR one vector artifact
Produces: one raster or vector artifact in the target CRS
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
class ReprojectParams(OperatorParams):
    """Parameters for reprojection."""

    target_crs: str = ""  # e.g. "EPSG:4326", "EPSG:32610"
    output_path: str = ""
    resampling: str = "nearest"  # nearest, bilinear, cubic, etc. (raster only)
    resolution: tuple[float, float] | None = None  # optional output resolution override


class ReprojectOperator:
    """Reprojects a raster or vector artifact to a new CRS."""

    @property
    def name(self) -> str:
        return "reproject"

    @property
    def spec(self) -> OperatorSpec:
        return OperatorSpec(
            input_types=(ArtifactType.RASTER, ArtifactType.VECTOR),
            output_type=ArtifactType.RASTER,  # output type matches input type
            min_inputs=1,
            max_inputs=1,
            resource_scale=ResourceScale.MEDIUM,
        )

    def validate_inputs(self, inputs: list[Artifact], params: OperatorParams) -> list[str]:
        errors = []

        if not inputs:
            errors.append("Exactly one input artifact required")
            return errors

        if len(inputs) > 1:
            errors.append(f"Expected 1 input, got {len(inputs)}")

        artifact = inputs[0]

        # Must be raster or vector
        if artifact.type not in (ArtifactType.RASTER, ArtifactType.VECTOR):
            errors.append(f"Input must be raster or vector, got {artifact.type.value}")

        # Must be materialized
        if not artifact.is_materialized:
            errors.append("Input artifact is not materialized (lazy handle)")

        # Must have a CRS to reproject FROM
        if artifact.spatial.crs is None:
            errors.append("Input artifact has no CRS — cannot reproject")

        # Params check
        if not isinstance(params, ReprojectParams):
            errors.append("Params must be ReprojectParams")
            return errors

        if not params.target_crs:
            errors.append("target_crs is required")

        if not params.output_path:
            errors.append("output_path is required")

        # No-op check: same CRS
        if artifact.spatial.crs and params.target_crs:
            if artifact.spatial.crs == params.target_crs:
                errors.append(f"Input CRS ({artifact.spatial.crs}) is already {params.target_crs}")

        return errors

    def execute(self, inputs: list[Artifact], params: OperatorParams) -> OperatorResult:
        if not isinstance(params, ReprojectParams):
            raise OperatorError(self.name, "Params must be ReprojectParams")

        artifact = inputs[0]

        if artifact.type == ArtifactType.RASTER:
            return self._reproject_raster(artifact, params)
        elif artifact.type == ArtifactType.VECTOR:
            return self._reproject_vector(artifact, params)
        else:
            raise OperatorError(
                self.name,
                f"Unsupported type: {artifact.type.value}",
                inputs=[artifact.id],
            )

    def declared_checks(self) -> list[str]:
        return [
            "crs_valid",
            "crs_matches_target",
            "extent_sane",
            "backing_accessible",
        ]

    # -----------------------------------------------------------------------
    # Raster reprojection
    # -----------------------------------------------------------------------

    def _reproject_raster(self, artifact: Artifact, params: ReprojectParams) -> OperatorResult:
        import rasterio
        from rasterio.crs import CRS
        from rasterio.warp import Resampling, calculate_default_transform, reproject

        input_path = artifact.backing.uri
        output_path = Path(params.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        target_crs = CRS.from_user_input(params.target_crs)

        # Map resampling string to enum
        resampling_map = {
            "nearest": Resampling.nearest,
            "bilinear": Resampling.bilinear,
            "cubic": Resampling.cubic,
            "cubic_spline": Resampling.cubic_spline,
            "lanczos": Resampling.lanczos,
            "average": Resampling.average,
            "mode": Resampling.mode,
        }
        resampling = resampling_map.get(params.resampling, Resampling.nearest)

        try:
            with rasterio.open(input_path) as src:
                if params.resolution:
                    res_x, res_y = params.resolution
                else:
                    res_x, res_y = None, None

                transform, width, height = calculate_default_transform(
                    src.crs,
                    target_crs,
                    src.width,
                    src.height,
                    *src.bounds,
                    resolution=(res_x, res_y) if res_x else None,
                )

                out_meta = src.meta.copy()
                out_meta.update(
                    {
                        "crs": target_crs,
                        "transform": transform,
                        "width": width,
                        "height": height,
                    }
                )

                with rasterio.open(output_path, "w", **out_meta) as dst:
                    for band_idx in range(1, src.count + 1):
                        reproject(
                            source=rasterio.band(src, band_idx),
                            destination=rasterio.band(dst, band_idx),
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=transform,
                            dst_crs=target_crs,
                            resampling=resampling,
                        )

        except Exception as e:
            raise OperatorError(
                self.name,
                f"Raster reprojection failed: {e}",
                inputs=[artifact.id],
            ) from e

        # Build output artifact with fresh spatial metadata from the actual output
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
                    crs=str(out_src.crs),
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
                        "target_crs": params.target_crs,
                        "resampling": params.resampling,
                        "resolution_override": params.resolution,
                        "source_crs": artifact.spatial.crs,
                    },
                ),
                metadata={
                    "driver": out_src.driver,
                    "dtype": str(out_src.dtypes[0]),
                },
            )

        checks = self._run_checks(output_artifact, artifact, params)
        return OperatorResult(artifact=output_artifact, checks=checks)

    # -----------------------------------------------------------------------
    # Vector reprojection
    # -----------------------------------------------------------------------

    def _reproject_vector(self, artifact: Artifact, params: ReprojectParams) -> OperatorResult:
        import fiona
        from fiona.crs import CRS as FionaCRS
        from fiona.transform import transform_geom

        input_path = artifact.backing.uri
        output_path = Path(params.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with fiona.open(input_path) as src:
                src_crs = src.crs
                dst_crs = FionaCRS.from_user_input(params.target_crs)

                out_schema = src.schema.copy()
                out_driver = src.driver

                with fiona.open(
                    output_path,
                    "w",
                    driver=out_driver,
                    crs=dst_crs,
                    schema=out_schema,
                ) as dst:
                    feature_count = 0
                    for feature in src:
                        transformed_geom = transform_geom(src_crs, dst_crs, feature["geometry"])
                        dst.write(
                            {
                                "geometry": transformed_geom,
                                "properties": feature["properties"],
                            }
                        )
                        feature_count += 1

        except Exception as e:
            raise OperatorError(
                self.name,
                f"Vector reprojection failed: {e}",
                inputs=[artifact.id],
            ) from e

        # Read back actual metadata from output
        with fiona.open(output_path) as out_src:
            out_bounds = out_src.bounds
            output_artifact = Artifact(
                type=ArtifactType.VECTOR,
                name=output_path.stem,
                backing=BackingStore(
                    kind=BackingStoreKind.LOCAL_FILE,
                    uri=str(output_path),
                    size_bytes=output_path.stat().st_size,
                    content_hash=content_hash(output_path),
                ),
                spatial=SpatialDescriptor(
                    crs=str(out_src.crs),
                    extent=(out_bounds[0], out_bounds[1], out_bounds[2], out_bounds[3]),
                    feature_count=feature_count,
                ),
                lineage=Lineage(
                    operation=self.name,
                    inputs=(artifact.id,),
                    params={
                        "target_crs": params.target_crs,
                        "source_crs": artifact.spatial.crs,
                    },
                ),
                metadata={
                    "driver": out_src.driver,
                    "schema": dict(out_src.schema),
                },
            )

        checks = self._run_checks(output_artifact, artifact, params)
        return OperatorResult(artifact=output_artifact, checks=checks)

    # -----------------------------------------------------------------------
    # Checks
    # -----------------------------------------------------------------------

    def _run_checks(
        self,
        output: Artifact,
        input_artifact: Artifact,
        params: ReprojectParams,
    ) -> list[CheckResult]:
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

        # CRS matches target
        if output.spatial.crs:
            # Normalize for comparison: both might be "EPSG:4326" or one might be
            # a full WKT. Simple string containment check on the EPSG code.
            target_epsg = params.target_crs.replace("EPSG:", "")
            output_crs_str = str(output.spatial.crs)
            if target_epsg in output_crs_str:
                results.append(
                    CheckResult(
                        check_name="crs_matches_target",
                        state=ValidationState.VALID,
                        message=f"Output CRS contains target EPSG:{target_epsg}",
                    )
                )
            else:
                results.append(
                    CheckResult(
                        check_name="crs_matches_target",
                        state=ValidationState.WARN,
                        message=f"Output CRS '{output_crs_str}' may not match target '{params.target_crs}'",
                    )
                )

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
